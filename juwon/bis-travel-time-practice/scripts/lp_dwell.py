"""회차 대기 역산 (LP): 차량 하루를 시각순으로 놓고, 모든 행의 도착 시각이 자기 arrive_hour 안에 들도록
  t_j = t0 + Σ_{i≤j} m_i + Σ(j 이전 미지수)
    운전 구간 m = 평소 시간(구간 중앙값) / 회차 구간 = 미지수 D (prior = 평소 회차 시간)
    빠진 행(같은 왕복에서 stop_seq 건너뜀) = 미지수 (prior = 직선거리 × 1.3 ÷ 20km/h)
    왕복이 바뀌는 곳(반대쪽 종점 대기, 행 없음) = 미지수 (prior 없음, ≥ 0)
  목표: Σ 시각 위반(초) + λ Σ |미지수 − prior|  → D = 하루 시각 패턴이 말해 주는 회차 대기
  입력(arrive_hour, 정류장 좌표)과 학습 날짜의 구간 중앙값만 사용.
"""
import numpy as np, pandas as pd
from scipy.optimize import linprog
from scipy import sparse

TRK = ['date', 'vehicle_id', 'direction', 'trip_no']


def order(df):
    d = df.copy()
    g = d.groupby(TRK)
    d['_ts'], d['_te'], d['_ss'] = g.arrive_hour.transform('min'), g.arrive_hour.transform('max'), g.stop_seq.transform('min')
    return d.sort_values(['date', 'vehicle_id', '_ts', '_te', '_ss', 'stop_seq'])


def solve(h, cum, pos, prior, lam):
    """h, cum: 행마다 시, 알려진 시간 누적. pos: 미지수가 영향 주기 시작하는 행 번호, prior, lam: 미지수별"""
    n, K = len(h), len(pos)
    nv = 1 + 3 * K + 2 * n
    iD, iup, ium, isp, ism = 1, 1 + K, 1 + 2 * K, 1 + 3 * K, 1 + 3 * K + n
    rows, cols, vals, b = [], [], [], []
    r = 0
    for j in range(n):
        kk = np.flatnonzero(pos <= j)
        for sign, sv, bb in [(-1, isp + j, cum[j] - 3600 * h[j]), (1, ism + j, 3600 * (h[j] + 1) - 1 - cum[j])]:
            rows += [r] * (2 + len(kk)); cols += [0] + list(iD + kk) + [sv]; vals += [sign] * (1 + len(kk)) + [-1]
            b.append(bb); r += 1
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(r, nv))
    ar = np.arange(K)
    Ae = sparse.csr_matrix((np.r_[np.ones(K), -np.ones(K), np.ones(K)], (np.r_[ar, ar, ar], np.r_[iD + ar, iup + ar, ium + ar])), shape=(K, nv))
    c = np.zeros(nv); c[iup:iup + K] = lam; c[ium:ium + K] = lam; c[isp:] = 1.0
    res = linprog(c, A_ub=A, b_ub=np.array(b), A_eq=Ae, b_eq=prior, bounds=[(None, None)] + [(0, None)] * (nv - 1), method='highs')
    if not res.success:
        return np.full(K, np.nan), np.full(n, np.nan)
    x = res.x; D = x[iD:iD + K]
    t = x[0] + cum + np.array([D[pos <= j].sum() for j in range(n)])
    return D, t


def hav(la1, lo1, la2, lo2):
    p = np.pi / 180
    a = np.sin((la2 - la1) * p / 2) ** 2 + np.cos(la1 * p) * np.cos(la2 * p) * np.sin((lo2 - lo1) * p / 2) ** 2
    return 12742000 * np.arcsin(np.sqrt(a))


def lp_dwell(df, med, lam=0.3, lam_gap=0.3, free=1e-3, depart=True):
    """반환: DataFrame(lp_D = 회차 행 역산 대기(나머지 NaN), lp_t = 행 출발 시각 추정(초), lp_min = 그 분(0~60)), df index 순서"""
    d = df.copy(); d['_m'] = med; d['_m'] = d._m.fillna(np.nanmedian(med))
    s = order(d)
    g = s.groupby(['date', 'vehicle_id'])
    first = (g.cumcount() == 0).values
    jump = (s.stop_seq - g.stop_seq.shift(1)).values
    disc = (g.to_stop.shift(1) != s.from_stop).values & ~first
    gap_in = disc & (jump > 1) & (jump < 30)            # 같은 왕복 안에서 행 빠짐
    gap_rt = disc & ~gap_in                              # 왕복 바뀜 (반대쪽 종점)
    dist = hav(g.to_lat.shift(1).values, g.to_lon.shift(1).values, s.from_lat.values, s.from_lon.values)
    gprior = np.nan_to_num(dist * 1.3 / (20 / 3.6))
    h = s.arrive_hour.values.astype(float); t = (s.is_turn2 == 1).values
    m = np.where(t, 0.0, s._m.values.astype(float))
    out = np.full(len(s), np.nan); tout = np.full(len(s), np.nan)
    vd = g.ngroup().values
    starts = np.flatnonzero(np.r_[True, np.diff(vd) != 0]); ends = np.r_[starts[1:], len(s)]
    for a, z in zip(starts, ends):
        tt = np.flatnonzero(t[a:z])
        if not len(tt):
            tt = np.array([], int)
        gi, gr = np.flatnonzero(gap_in[a:z]), np.flatnonzero(gap_rt[a:z])
        pos = np.r_[tt + (1 if depart else 0), gi, gr]   # depart: arrive_hour = 구간 시작 시각 → 회차 D는 다음 행부터 영향
        prior = np.r_[s._m.values[a:z][tt], gprior[a:z][gi], np.zeros(len(gr))]
        lam_v = np.r_[np.full(len(tt), lam), np.full(len(gi), lam_gap), np.full(len(gr), free)]
        cm = np.cumsum(m[a:z]); cm = np.r_[0, cm[:-1]] if depart else cm
        x, tj = solve(h[a:z], cm, pos.astype(int), prior, lam_v)
        out[a + tt] = x[:len(tt)]; tout[a:z] = tj
    r = pd.DataFrame({'lp_D': out, 'lp_t': tout}, index=s.index)
    r['lp_min'] = (r.lp_t - s.arrive_hour * 3600).clip(0, 3599) / 60
    return r.loc[df.index]
