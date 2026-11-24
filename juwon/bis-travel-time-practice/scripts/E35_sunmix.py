"""E35 일요일 행만 '토요일처럼'과 '평일처럼' 예측을 반반 섞기 (E32 모델 그대로)

실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E35_sunmix.py practice   (또는 final)
결과: submissions/{practice|final}_E35_sunmix.csv (약 2분)
주의: 노트북 cv_feature_experiments.ipynb 의 셀(2,4,6,8,10,27,29,41번)을 순서대로 실행해서 피처·모델 함수를 가져옴.
      노트북 셀 순서가 바뀌면 이 스크립트도 고쳐야 함.
"""
import json, os, sys, time
import numpy as np, pandas as pd
os.chdir('/Users/kimjuwon/Desktop/에모팀플/bis-travel-time-practice/notebooks')
PH = sys.argv[1]
nb = json.load(open('cv_feature_experiments.ipynb')); G = {}
for i in [2, 4, 6, 8, 10, 27, 29, 41]:
    src = ''.join(nb['cells'][i]['source'])
    if i == 2: src = __import__('re').sub(r"PHASE = '\w+'", f"PHASE = '{PH}'", src, count=1)
    exec(src, G)
TRAIN, TEST, test, to_X = G['TRAIN'], G['TEST'], G['test'], G['to_X']
from catboost import CatBoostRegressor
import lightgbm as lgb
F = G['feats_of'](['base', 'seg', 'time', 'trip'])
LGB = {**G['PARAMS'], 'min_child_samples': 10, 'learning_rate': 0.02, 'n_estimators': 1500}
CB = dict(loss_function='MAE', iterations=2000, learning_rate=0.08, depth=10, l2_leaf_reg=3, random_seed=0, verbose=0, thread_count=-1)
t0 = time.time()
ml = lgb.LGBMRegressor(**LGB).fit(to_X(TRAIN, F), TRAIN.tt_sec)
mc = CatBoostRegressor(**CB, cat_features=['route_id', 'seg_id']).fit(TRAIN[F], TRAIN.tt_sec)
pred = lambda X: 0.8 * ml.predict(to_X(X, F)) + 0.2 * mc.predict(X[F])
sun = pd.to_datetime(TEST.date).dt.day_of_week.values == 6
print(f'[{PH}] 학습 {time.time()-t0:.0f}초 | 일요일 행 {sun.sum()}', flush=True)
base = pred(TEST)
S = TEST.copy(); S['is_weekend'] = 1; S['interval'] = S.interval_sat     # 토요일처럼
W = TEST.copy(); W['is_weekend'] = 0; W['interval'] = W.interval_wd      # 평일처럼
pS, pW = pred(S), pred(W)
V = {'E32 그대로 (주말, 일요일 배차)': base,
     '토요일처럼 (주말, 토 배차)': np.where(sun, pS, base),
     '평일처럼 (평일, 평일 배차)': np.where(sun, pW, base),
     '반반: 토요일처럼 + 평일처럼': np.where(sun, 0.5 * pS + 0.5 * pW, base),
     '반반: E32 + 평일처럼 (참고)': np.where(sun, 0.5 * base + 0.5 * pW, base)}
print(f'[{PH}] 일요일 예측 평균: E32 {base[sun].mean():.1f} / 토처럼 {pS[sun].mean():.1f} / 평일처럼 {pW[sun].mean():.1f}', flush=True)
has = 'tt_sec' in test.columns
for name, p in V.items():
    sub = pd.DataFrame({'row_id': TEST.row_id.values, 'tt_sec': np.round(p, 1)}).set_index('row_id').loc[test.row_id].reset_index()
    if has:
        y = TEST.tt_sec.values; a = np.abs(y - np.round(p, 1)); res = (y - np.round(p, 1))[sun]
        print(f'[{PH}] {name:28s} test {a.mean():.3f} | 일 {a[sun].mean():.3f} | 일 잔차중앙값 {np.median(res):+.2f}', flush=True)
    if name.startswith('반반: 토요일처럼'):
        assert len(sub) == len(test) and sub.tt_sec.notna().all() and not sub.row_id.duplicated().any()
        sub.to_csv(f'../submissions/{PH}_E35_sunmix.csv', index=False); print(f'[{PH}] 저장 ../submissions/{PH}_E35_sunmix.csv', flush=True)
if not has:
    prev = pd.read_csv(f'../submissions/{PH}_E32.csv').set_index('row_id').tt_sec
    print(f'[{PH}] 검산: 그대로 버전 = 제출한 final_E32.csv?', bool((prev - pd.Series(np.round(base, 1), index=TEST.row_id.values)).abs().max() < 1e-9))
