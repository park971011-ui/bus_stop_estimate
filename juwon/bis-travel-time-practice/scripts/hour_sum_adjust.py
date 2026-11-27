"""E106 시간 합 후처리 (E93 다시): 차량 하루에서 시 h의 행 묶음(block)이 앞 시 h-1, 뒤 시 h+1과 끊김 없이 이어지면
  block 합 ≈ 3600 + (x_last − x_prev)/2  (x_prev = h-1 마지막 행, x_last = block 마지막 행)
  남는 차이 R = 목표 − 예측 합 의 frac 만큼을 block 행에 나눔: 회차 행이 있으면 turn_share는 회차 행에, 나머지는 예측값^pw 비율로
정답 안 씀 (입력: arrive_hour, 정류장 연결, 예측값)
"""
import numpy as np, pandas as pd
from lp_dwell import order


def hour_sum_adjust(df, pred, frac=0.25, turn_share=1.0, pw=1.0, max_ratio=0.3):
    d = df.copy(); d['_p'] = np.asarray(pred, float)
    s = order(d)
    g = s.groupby(['date', 'vehicle_id'])
    first = (g.cumcount() == 0).values
    brk = first | (g.to_stop.shift(1) != s.from_stop).values      # 이 행 앞에서 끊김
    p = s._p.values.copy(); h = s.arrive_hour.values; turn = (s.is_turn2 == 1).values
    vd = g.ngroup().values
    key = vd * 100 + h
    starts = np.flatnonzero(np.r_[True, np.diff(key) != 0]); ends = np.r_[starts[1:], len(s)]
    adj = p.copy(); nb = 0
    for i, (a, z) in enumerate(zip(starts, ends)):
        if a == 0 or z >= len(s): continue
        if vd[a - 1] != vd[a] or vd[z] != vd[a]: continue
        if h[a - 1] != h[a] - 1 or h[z] != h[a] + 1: continue
        if brk[a:z + 1].any(): continue                             # block 안·양 끝 끊김 없음
        target = 3600 + (p[z - 1] - p[a - 1]) / 2
        R = target - p[a:z].sum()
        if abs(R) > max_ratio * 3600: continue
        R *= frac; nb += 1
        tv = turn[a:z]
        w = np.maximum(p[a:z], 1) ** pw
        if tv.any() and turn_share > 0:
            wt = np.where(tv, w, 0); wn = np.where(tv, 0, w)
            add = turn_share * R * wt / wt.sum() + (1 - turn_share) * R * wn / wn.sum()
        else:
            add = R * w / w.sum()
        adj[a:z] = np.maximum(p[a:z] + add, 0)
    return pd.Series(adj, index=s.index).loc[df.index].values, nb
