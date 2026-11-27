"""시각 제약 가우시안 보정 (GLS): 차량 하루를 시각순으로 놓고
  행 시간 x_i ~ N(p_i, (k·p_i^β)²)  (p = 모델 예측), 회차 행은 σ를 kt배, 행 빠진 곳/왕복 바뀜은 미지수 g (넓은 prior)
  정각 T = 3600·h 는 연속한 두 행 출발 사이 [S_j, S_{j+1}) 어딘가 → S_j + 0.5·(S_{j+1}-S_j) ≈ T, 오차 σ = 길이/√12
  선형 가우시안이라 사후 평균이 닫힌 해. 정답(tt_sec) 안 씀: 입력은 arrive_hour, 정류장, 예측값뿐.
"""
import numpy as np, pandas as pd
from lp_dwell import order, hav


def gls_adjust(df, pred, k=0.35, beta=1.0, kt=3.0, gap_sd=600.0, sig_floor=8.0, cross_scale=1.0, mean_ratio=1.0, turn_ratio=1.0):
    """pred = 중앙값 예측. prior 평균 = pred×mean_ratio(회차 행 turn_ratio). 반환 = pred + (사후 평균 − prior 평균)"""
    d = df.copy(); d['_p'] = np.asarray(pred, float)
    s = order(d)
    g = s.groupby(['date', 'vehicle_id'])
    first = (g.cumcount() == 0).values
    jump = (s.stop_seq - g.stop_seq.shift(1)).values
    disc0 = (g.to_stop.shift(1) != s.from_stop).values & ~first
    bigjump = (g.arrive_hour.diff().values >= 2) & ~first
    disc = disc0 | bigjump
    gap_in = disc0 & (jump > 1) & (jump < 30) & ~bigjump
    dist = hav(g.to_lat.shift(1).values, g.to_lon.shift(1).values, s.from_lat.values, s.from_lon.values)
    gprior_in = np.nan_to_num(dist * 1.3 / (20 / 3.6))
    h = s.arrive_hour.values.astype(float)
    p0 = s._p.values
    turn = (s.is_turn2 == 1).values
    p = p0 * np.where(turn, turn_ratio, mean_ratio)
    sig = np.maximum(k * np.maximum(p, 1) ** beta * np.where(turn, kt, 1.0), sig_floor)
    out = p.copy()  # 사후 평균 (prior 평균 기준)
    vd = g.ngroup().values
    starts = np.flatnonzero(np.r_[True, np.diff(vd) != 0]); ends = np.r_[starts[1:], len(s)]
    for a, z in zip(starts, ends):
        n = z - a
        if n < 2:
            continue
        gi = np.flatnonzero(disc[a:z])            # 행 j 앞에 빈 시간 g
        m = len(gi)
        # 변수: [t0, x_0..x_{n-1}, g_0..g_{m-1}]
        nv = 1 + n + m
        # S_j = t0 + Σ_{i<j} x_i + Σ_{gaps at idx<=j} g
        Sm = np.zeros((n, nv)); Sm[:, 0] = 1
        Sm[:, 1:1 + n] = np.tri(n, n, -1)
        for q, jj in enumerate(gi):
            Sm[jj:, 1 + n + q] = 1
        pv = p[a:z]; hv = h[a:z]
        gp = np.where(gap_in[a:z][gi], gprior_in[a:z][gi], np.where(bigjump[a:z][gi], 0.0, 300.0)) if m else np.zeros(0)
        gs = np.where(gap_in[a:z][gi], np.maximum(gprior_in[a:z][gi], 60.0), np.where(bigjump[a:z][gi], 1e5, gap_sd)) if m else np.zeros(0)
        # 관측: 연속 행 j, j+1 사이 정각
        rows, T, w = [], [], []
        for j in range(n - 1):
            if hv[j + 1] > hv[j]:
                ln = pv[j] + (gp[gi == j + 1].sum() if m else 0)
                sd_len = np.sqrt(sig[a + j] ** 2 + ((gs[gi == j + 1] ** 2).sum() if m else 0))
                for hh in np.arange(hv[j] + 1, hv[j + 1] + 1):
                    rows.append(0.5 * (Sm[j] + Sm[j + 1])); T.append(3600 * hh)
                    w.append(1.0 / (cross_scale * (ln ** 2 / 12 + 1.0) + 0.25 * sd_len ** 2 * 0))
        if not rows:
            continue
        A = np.array(rows); T = np.array(T); w = np.array(w)
        prec = np.zeros(nv); mu = np.zeros(nv)
        prec[0] = 1e-12
        prec[1:1 + n] = 1 / sig[a:z] ** 2; mu[1:1 + n] = pv
        if m:
            prec[1 + n:] = 1 / gs ** 2; mu[1 + n:] = gp
        # t0 prior: 첫 행 시 중앙
        M = A.T @ (A * w[:, None]) + np.diag(prec)
        rhs = A.T @ (w * T) + prec * mu
        # t0 초기값을 첫 행 시의 중간으로 약하게
        M[0, 0] += 1e-8; rhs[0] += 1e-8 * (3600 * hv[0] + 1800)
        try:
            zsol = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            continue
        out[a:z] = zsol[1:1 + n]
    res = pd.Series(p0 + (out - p), index=s.index).loc[df.index].values
    return res
