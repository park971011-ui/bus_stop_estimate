"""예측 후처리 (LP): 차량 하루를 시각순으로 놓고, 예측값을 최소한(L1)으로만 고쳐서 모든 행의 출발 시각이 자기 arrive_hour 안에 들게 함
  변수: 사슬 시작 t0, 행마다 고친 값 x_i ≥ 0, |x_i − p_i|, 빠진 행/왕복 바뀜 미지수, 시각 위반 slack
  목표: Σ w_i|x_i − p_i| + λ_gap Σ|gap − prior| + W Σ slack
  출발 시각 t_j = t0 + Σ_{i<j} x_i + Σ(j 이전 gap)   (arrive_hour = 출발 시각의 시, E87에서 확인)
  입력(arrive_hour, 좌표)과 모델 예측만 사용.
"""
import numpy as np, pandas as pd
from scipy.optimize import linprog
from scipy import sparse
from lp_dwell import order, hav


def solve(h, p, w, gpos, gprior, glam, W):
    n, K = len(h), len(gpos)
    # 변수 순서: t0 | x(n) | e+(n) e-(n) | g(K) | g+(K) g-(K) | s+(n) s-(n)
    ix, iep, iem = 1, 1 + n, 1 + 2 * n
    ig, igp, igm = 1 + 3 * n, 1 + 3 * n + K, 1 + 3 * n + 2 * K
    isp, ism = 1 + 3 * n + 3 * K, 1 + 4 * n + 3 * K
    nv = 1 + 5 * n + 3 * K
    # 부등식: t_j = t0 + Σ_{i<j} x_i + Σ_{gpos≤j} g
    #  -t_j - s+_j ≤ -3600h_j ;  t_j - s-_j ≤ 3600h_j + 3599
    R, C, V, b = [], [], [], []
    r = 0
    for j in range(n):
        xs = list(range(ix, ix + j)); gs = list(ig + np.flatnonzero(gpos <= j))
        for sign, sv, bb in [(-1, isp + j, -3600 * h[j]), (1, ism + j, 3600 * h[j] + 3599)]:
            cols = [0] + xs + gs + [sv]
            R += [r] * len(cols); C += cols; V += [sign] * (len(cols) - 1) + [-1]; b.append(bb); r += 1
    A = sparse.csr_matrix((V, (R, C)), shape=(r, nv))
    an, ak = np.arange(n), np.arange(K)
    Ae = sparse.csr_matrix((np.r_[np.ones(n), -np.ones(n), np.ones(n), np.ones(K), -np.ones(K), np.ones(K)],
                            (np.r_[an, an, an, n + ak, n + ak, n + ak],
                             np.r_[ix + an, iep + an, iem + an, ig + ak, igp + ak, igm + ak])), shape=(n + K, nv))
    c = np.zeros(nv); c[iep:iep + n] = w; c[iem:iem + n] = w; c[igp:igp + K] = glam; c[igm:igm + K] = glam; c[isp:] = W
    res = linprog(c, A_ub=A, b_ub=np.array(b, float), A_eq=Ae, b_eq=np.r_[p, gprior],
                  bounds=[(None, None)] + [(0, None)] * (nv - 1), method='highs')
    return res.x[ix:ix + n] if res.success else p


def lp_adjust(df, pred, w=None, lam_gap=0.3, free=1e-3, W=5.0):
    d = df.copy(); d['_p'] = pred; d['_w'] = 1.0 if w is None else w
    s = order(d)
    g = s.groupby(['date', 'vehicle_id'])
    first = (g.cumcount() == 0).values
    jump = (s.stop_seq - g.stop_seq.shift(1)).values
    disc = (g.to_stop.shift(1) != s.from_stop).values & ~first
    gap_in = disc & (jump > 1) & (jump < 30)
    gap_rt = disc & ~gap_in
    dist = hav(g.to_lat.shift(1).values, g.to_lon.shift(1).values, s.from_lat.values, s.from_lon.values)
    gprior = np.nan_to_num(dist * 1.3 / (20 / 3.6))
    h = s.arrive_hour.values.astype(float); p = s._p.values.astype(float); ww = s._w.values.astype(float)
    out = p.copy()
    vd = g.ngroup().values
    starts = np.flatnonzero(np.r_[True, np.diff(vd) != 0]); ends = np.r_[starts[1:], len(s)]
    for a, z in zip(starts, ends):
        gi, gr = np.flatnonzero(gap_in[a:z]), np.flatnonzero(gap_rt[a:z])
        gpos = np.r_[gi, gr].astype(int)
        out[a:z] = solve(h[a:z], p[a:z], ww[a:z], gpos, np.r_[gprior[a:z][gi], np.zeros(len(gr))],
                         np.r_[np.full(len(gi), lam_gap), np.full(len(gr), free)], W)
    return pd.Series(out, index=s.index).loc[df.index].values
