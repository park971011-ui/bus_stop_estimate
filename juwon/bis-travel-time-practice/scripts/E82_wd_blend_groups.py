"""E82 목·금 섞는 비율을 그룹별로 (기본 모델 × (1−w) + 평일 전용 모델 × w, 지금은 모든 행 w = 0.5)
  피처: E79와 같은 45개 (E74 조합 + 정각 넘김). 기본 = LGB 0.8 + CB 0.2 (4일 중 검증일 뺀 3일 학습), 평일 전용 = LGB (월·화·수 중 검증일 뺀 2일)
  그룹: (a) 전체 하나 (b) 노선별 (c) 시간대 (출근 7~9 / 퇴근 17~19 / 그 외) (d) 회차 행 / 나머지
  과적합 방지: 검증일 d의 비율은 나머지 두 평일로만 고름 (w = 0, 0.1, …, 1) → d로 채점, 3번 반복
  채택 기준(미리 고정): 0.5 고정보다 평일 전체 MAE가 낮고 3일 중 2일 이상 좋아짐
실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E82_wd_blend_groups.py final
"""
import sys, time, json
import numpy as np, pandas as pd, lightgbm as lgb

ROOT = '/Users/kimjuwon/Desktop/에모팀플/bis-travel-time-practice'
src = open(f'{ROOT}/scripts/E79_cross_wd.py').read().split('\nt0 = time.time()')[0]
exec(src)                                   # PH, TRAIN, prep, FE, LGB, CB, CatBoostRegressor, to_X, dow
assert PH == 'final'

t0 = time.time()
WD = dow(TRAIN) < 5
pB = pd.Series(np.nan, index=TRAIN.index); pW = pd.Series(np.nan, index=TRAIN.index); TURNF = pd.Series(0, index=TRAIN.index)
for d in sorted(TRAIN[WD].date.unique()):
    m = (TRAIN.date == d).values
    f, v = prep(TRAIN[~m], TRAIN[m])
    ml = lgb.LGBMRegressor(**LGB).fit(to_X(f, FE), f.tt_sec)
    mc = CatBoostRegressor(**CB, cat_features=['route_id', 'seg_id']).fit(f[FE], f.tt_sec)
    pB[v.index] = 0.8 * ml.predict(to_X(v, FE)) + 0.2 * mc.predict(v[FE])
    TURNF[v.index] = v.is_turn2.values
    fw, vw = prep(TRAIN[WD & ~m], TRAIN[m])
    pW[vw.index] = lgb.LGBMRegressor(**LGB).fit(to_X(fw, FE), fw.tt_sec).predict(to_X(vw, FE))
    print(f'{d} 끝 ({time.time() - t0:.0f}초)', flush=True)

D = TRAIN[WD].copy()
D['pB'], D['pW'], D['turn'] = pB[WD].values, pW[WD].values, TURNF[WD].values
D.to_parquet(f'{ROOT}/experiments/E82_oof.parquet')
y = D.tt_sec.values
h = D.arrive_hour.values
SCHEMES = {
    '(a) 전체 하나': np.zeros(len(D), int).astype(str),
    '(b) 노선별': D.route_nm.values.astype(str),
    '(c) 시간대': np.where(np.isin(h, [7, 8, 9]), '출근', np.where(np.isin(h, [17, 18, 19]), '퇴근', '그 외')),
    '(d) 회차 행': np.where(D.turn.values == 1, '회차', '나머지'),
}
WS = np.round(np.arange(0, 1.01, 0.1), 1)
days = sorted(D.date.unique())
mae = lambda m, w: np.abs(y[m] - ((1 - w) * D.pB.values[m] + w * D.pW.values[m])).mean()
base = {d: mae((D.date == d).values, 0.5) for d in days}
nall = len(D)
tot0 = sum(base[d] * (D.date == d).sum() for d in days) / nall
print(f'\n0.5 고정: 평일 {tot0:.3f} | ' + ' '.join(f'{d[5:]} {base[d]:.3f}' for d in days), flush=True)
res = {}
for name, grp in SCHEMES.items():
    out = {}
    picks = {}
    for d in days:
        te_m = (D.date == d).values; tr_m = ~te_m
        err = 0.0
        pk = {}
        for g in np.unique(grp):
            gm_tr, gm_te = tr_m & (grp == g), te_m & (grp == g)
            if gm_tr.sum() == 0 or gm_te.sum() == 0:
                continue
            w = WS[np.argmin([mae(gm_tr, w) for w in WS])]; pk[g] = w
            err += np.abs(y[gm_te] - ((1 - w) * D.pB.values[gm_te] + w * D.pW.values[gm_te])).sum()
        out[d] = err / te_m.sum(); picks[d] = pk
    tot = sum(out[d] * (D.date == d).sum() for d in days) / nall
    better = sum(out[d] < base[d] for d in days)
    res[name] = {'tot': tot, 'better': better, 'days': out, 'picks': picks}
    full = {g: float(WS[np.argmin([mae(grp == g, w) for w in WS])]) for g in np.unique(grp)}
    print(f'{name:10s} 평일 {tot:.3f} ({tot - tot0:+.3f}) | ' + ' '.join(f'{d[5:]} {out[d]:.3f}' for d in days) +
          f' | 좋아진 날 {better}/3 | 3일 전체로 고른 비율 {full}', flush=True)
ok = {k: v for k, v in res.items() if v['tot'] < tot0 and v['better'] >= 2}
best = min(ok, key=lambda k: ok[k]['tot']) if ok else None
print(f'→ {"채택: " + best if best else "채택 안 함 (0.5 고정 유지)"}', flush=True)
json.dump({'base': tot0, 'best': best, 'res': {k: {'tot': v['tot'], 'better': v['better']} for k, v in res.items()}},
          open(f'{ROOT}/experiments/E82_choice.json', 'w'), ensure_ascii=False)
