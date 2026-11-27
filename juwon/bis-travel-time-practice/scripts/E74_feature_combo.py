"""E74 피처 조합 찾기 (대회 train 날짜별 4-fold, 목·금 모델 기준)
  기준 = E41 피처 (base, seg, time, trip, 회차 힌트 H) = 35개
  1단계 빼보기: 기준에서 묶음 하나씩 뺌
  2단계 더해보기: 기준에 안 쓰는 묶음 하나씩 더함 (te, busy, station, peak2, wkhour, turn, struct, share, center, geo, tolast, turn_est, est_min,
               새 피처 blk(차량·시간 칸 평소 시간 합 비율), rhour(같은 날·노선·시간 버스들의 비율 중앙값))
  1·2단계는 빠른 설정(lr 0.05, 600개)으로 순위만 봄 → 좋아진 것(기준보다 예상 0.01 이상 낮음)만 모아
  3단계 원래 설정(lr 0.02, 1500개)으로 기준(25.227)과 비교. 채택 기준: 예상 점수 낮고 평일 3일 중 2일 이상 좋아짐
실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E74_feature_combo.py
"""
import sys, time, json
import numpy as np, pandas as pd, lightgbm as lgb

ROOT = '/Users/kimjuwon/Desktop/에모팀플/bis-travel-time-practice'
src = open(f'{ROOT}/scripts/E72_all_features.py').read().split('\nt0 = time.time()')[0]
exec(src)                                   # TA 준비 함수, add_te, add_est, add_minute, seg_med, G, H, LGB, cv_of ...

FG = G['FEATS']
CUR = {'base': FG['base'], 'seg': FG['seg'], 'time': FG['time'], 'trip': FG['trip'], 'hint': H}
ADD = {k: FG[k] for k in ['te', 'busy', 'station', 'peak2', 'wkhour', 'turn', 'struct', 'share', 'center', 'geo', 'tolast']}
ADD['turn_est'] = ['turn_est']; ADD['minute'] = ['est_min', 'est_time']
ADD['blk'] = ['blk_usual', 'blk_full', 'blk_ratio', 'blk_ratio_prev', 'blk_ratio_next']
ADD['rhour'] = ['rhour_ratio', 'rhour_n', 'rday_ratio']


def add_blk(df, med):
    """새 피처: 차량·시간 칸의 평소 시간 합으로 '그 시간 얼마나 빨리 달렸나' (정답 없이 시 기록 + 다른 날 평소 시간만)"""
    d = df.copy(); d['_m'] = med
    d['_m'] = d._m.fillna(np.nanmedian(med))
    K = ['date', 'vehicle_id', 'arrive_hour']
    hs = d.groupby(K)._m.sum()
    idx = hs.index
    k = lambda s: pd.MultiIndex.from_arrays([d.date, d.vehicle_id, d.arrive_hour + s])
    full = (k(-1).isin(idx) & k(1).isin(idx))                          # 앞뒤 시간 기록이 다 있음 = 한 시간을 꽉 채워 달림
    d['blk_usual'] = hs.reindex(k(0)).values
    d['blk_full'] = full.astype(int)
    d['blk_ratio'] = np.where(full, 3574.0 / d.blk_usual, np.nan)       # 1보다 크면 평소보다 느림
    fr = pd.Series(np.where(full, 3574.0 / d.blk_usual, np.nan), index=pd.MultiIndex.from_arrays([d.date, d.vehicle_id, d.arrive_hour]))
    fr = fr[~fr.index.duplicated()]
    d['blk_ratio_prev'] = fr.reindex(k(-1)).values
    d['blk_ratio_next'] = fr.reindex(k(1)).values
    # 같은 날·노선·시간 버스들의 중앙값 = 그 시간 노선 전체가 얼마나 막혔나
    vb = d[full].groupby(K + ['route_id']).blk_ratio.first().reset_index()
    rh = vb.groupby(['date', 'route_id', 'arrive_hour']).blk_ratio.agg(rhour_ratio='median', rhour_n='size')
    d = d.join(rh, on=['date', 'route_id', 'arrive_hour'])
    d = d.join(vb.groupby(['date', 'route_id']).blk_ratio.median().rename('rday_ratio'), on=['date', 'route_id'])
    return d.drop(columns='_m')


FAST = {**LGB, 'learning_rate': 0.05, 'n_estimators': 600}

t0 = time.time()
TA = TR.copy()
folds = []
for dd in DATES:                            # fold마다 정답 쓰는 피처를 한 번만 계산해 둠
    fm = (TA.date != dd).values; f, v = TA[fm], TA[~fm]
    f, v = G['add_te'](f, v)
    mf = np.full(len(f), np.nan)
    for d2 in f.date.unique():
        m = (f.date == d2).values; mf[m] = seg_med(f[~m], f[m])
    mv = seg_med(f, v)
    f, v = add_est(f, mf), add_est(v, mv)
    f, v = add_minute(f, mf), add_minute(v, mv)
    f, v = add_blk(f, mf), add_blk(v, mv)
    folds.append((f, v))
print(f'fold 준비 {time.time() - t0:.0f}초', flush=True)


def cv(feats, params):
    o = pd.Series(np.nan, index=TA.index)
    for f, v in folds:
        o[v.index] = lgb.LGBMRegressor(**params).fit(to_X(f, feats), f.tt_sec).predict(to_X(v, feats))
    r = cv_of(o)
    return {k: r[k] for k in ['월', '화', '수', '토', '예상']}


base_f = sum(CUR.values(), [])
rows = []
b = cv(base_f, FAST); rows.append({'변경': '기준 (E41 피처)', **b}); print(f"기준 예상 {b['예상']:.3f} ({time.time() - t0:.0f}초)", flush=True)
for k in CUR:
    r = cv([c for c in base_f if c not in CUR[k]], FAST); rows.append({'변경': f'− {k}', **r})
    print(f"− {k:9s} 예상 {r['예상']:.3f} ({r['예상'] - b['예상']:+.3f}) ({time.time() - t0:.0f}초)", flush=True)
for k in ADD:
    r = cv(base_f + ADD[k], FAST); rows.append({'변경': f'+ {k}', **r})
    print(f"+ {k:9s} 예상 {r['예상']:.3f} ({r['예상'] - b['예상']:+.3f}) ({time.time() - t0:.0f}초)", flush=True)
T = pd.DataFrame(rows).set_index('변경'); T['차이'] = T['예상'] - b['예상']
T.round(3).to_csv(f'{ROOT}/experiments/E74_screen.csv')

good = [i for i in T.index if i != '기준 (E41 피처)' and T.loc[i, '차이'] <= -0.01]
combo = list(base_f)
for i in good:
    k = i[2:]
    combo = [c for c in combo if c not in CUR[k]] if i.startswith('−') else combo + ADD[k]
print(f'좋아진 변경: {good}', flush=True)
if good:
    r0 = cv(base_f, LGB); r1 = cv(combo, LGB)
    wd_better = sum(r1[d] < r0[d] for d in ['월', '화', '수'])
    ok = r1['예상'] < r0['예상'] and wd_better >= 2
    print(f"[확인] 기준 {r0['예상']:.3f} (월 {r0['월']:.3f} 화 {r0['화']:.3f} 수 {r0['수']:.3f} 토 {r0['토']:.3f})", flush=True)
    print(f"[확인] 조합 {r1['예상']:.3f} (월 {r1['월']:.3f} 화 {r1['화']:.3f} 수 {r1['수']:.3f} 토 {r1['토']:.3f}) | 평일 좋아진 날 {wd_better}/3 → {'채택' if ok else '채택 안 함'}", flush=True)
    json.dump({'good': good, 'n_feats': len(combo), 'base': r0, 'combo': r1, 'ok': bool(ok)}, open(f'{ROOT}/experiments/E74_choice.json', 'w'), ensure_ascii=False)
else:
    print('좋아진 변경 없음 → 지금 조합 유지', flush=True)
