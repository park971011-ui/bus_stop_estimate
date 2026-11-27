"""E99 일요일 모델(E73) + 일요일 회차 피처 — E94(후처리 배율)를 학습 안으로
  sun_turn    = is_sunday × is_turn2
  sun_turn_iv = sun_turn × (interval_sun / interval_wd)
  검증: E73과 같은 실습 노선 하나 빼기(LightGBM만). 기준 E73 PX(27.340, E73_loro.parquet 재사용)
        참고: E73 + E94 배율(나머지 3노선으로 정한 L1 배율) 26.860 / E99 + E94 배율도 같은 방식으로
  채택 기준(미리 고정): E73보다 낮고 4개 노선 중 3개 이상 개선. E94 배율과 겹치는지도 확인
실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E99_sunday_turn_feat.py
"""
import sys
ROOT = '/Users/kimjuwon/Desktop/에모팀플/bis-travel-time-practice'
src = open(f'{ROOT}/scripts/E73_sunday_px.py').read().split('\nt0 = time.time()')[0]
sys.argv = [sys.argv[0], 'cv']
exec(src)
NEW = ['sun_turn', 'sun_turn_iv']


def add_new(d):
    d = d.copy()
    d['sun_turn'] = ((dow(d) == 6) & (d.is_turn2 == 1)).astype(int)
    d['sun_turn_iv'] = d.sun_turn * d.interval_sun / d.interval_wd
    return d


FTR, PTR, PTE = add_new(FTR), add_new(PTR), add_new(PTE)
E73 = pd.read_parquet(f'{ROOT}/experiments/E73_loro.parquet').set_index('row_id')


def l1k(y, p):
    r = y / p; o = np.argsort(r); c = np.cumsum(p[o]); return r[o][np.searchsorted(c, c[-1] / 2)]


def scale_nested(L, col):
    """빠진 노선마다 나머지 3노선 회차 행으로 정한 배율을 그 노선 회차 행에 곱함 (E94와 같음)"""
    out = L[col].values.copy()
    for r in L.route.unique():
        o = L[(L.route != r) & (L.turn == 1)]
        m = ((L.route == r) & (L.turn == 1)).values
        out[m] *= l1k(o.y.values, o[col].values)
    return out


t0 = time.time()
rows = []
for r in sorted(PTE.route_nm.unique()):
    m = (PTE.route_nm == r).values
    P_all = pd.concat([PTR, PTE[~m]])
    tr = with_px([FTR, PTR, PTE[~m]], P_all)
    v = with_px([PTE[m & (dow(PTE) == 6)]], P_all)
    p = lgb.LGBMRegressor(**LGB).fit(Xf(tr, FX + NEW), tr.tt_sec).predict(Xf(v, FX + NEW))
    L = pd.DataFrame({'row_id': v.row_id.values, 'route': r, 'y': v.tt_sec.values, 'turn': v.is_turn2.values, 'E99': p})
    L['E73'] = E73.loc[L.row_id, 'PX'].values
    print(f'[검증] 노선 {r}: E73 {(L.y - L.E73).abs().mean():.3f} → E99 {(L.y - L.E99).abs().mean():.3f} | 회차 행 {L.turn.sum()} ({time.time() - t0:.0f}초)', flush=True)
    rows.append(L)
L = pd.concat(rows)
L['E73_s'] = scale_nested(L, 'E73'); L['E99_s'] = scale_nested(L, 'E99')
L.to_parquet(f'{ROOT}/experiments/E99_loro.parquet')
cols = ['E73', 'E73_s', 'E99', 'E99_s']
t = pd.DataFrame({c: L.groupby('route').apply(lambda g: (g.y - g[c]).abs().mean()) for c in cols})
t.loc['일요일 전체'] = [(L.y - L[c]).abs().mean() for c in cols]
t.loc['회차 행'] = [(L.y - L[c])[L.turn == 1].abs().mean() for c in cols]
t.to_csv(f'{ROOT}/experiments/E99_cv.csv')
print(t.round(3).to_string())
e = t.drop(['일요일 전체', '회차 행'])
print(f"[판정] E99 vs E73: 좋아진 노선 {(e.E99 < e.E73).sum()}/4 | E99_s vs E73_s(E94): {(e.E99_s < e.E73_s).sum()}/4")
