"""E36 seed 5개 섞기 + 일요일 반반 (LightGBM·CatBoost 각각 seed 0~4)

실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E36_seed5_sunmix.py practice   (또는 final)
결과: submissions/{practice|final}_E36_seed5_sunmix.csv (약 6~7분)
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
CB = dict(loss_function='MAE', iterations=2000, learning_rate=0.08, depth=10, l2_leaf_reg=3, verbose=0, thread_count=-1)
sun = pd.to_datetime(TEST.date).dt.day_of_week.values == 6
S = TEST.copy(); S['is_weekend'] = 1; S['interval'] = S.interval_sat
W = TEST.copy(); W['is_weekend'] = 0; W['interval'] = W.interval_wd
P = {k: [] for k in ['lb', 'ls', 'lw', 'cb', 'cs', 'cw']}
t0 = time.time()
for seed in range(5):
    ml = lgb.LGBMRegressor(**{**LGB, 'random_state': seed}).fit(to_X(TRAIN, F), TRAIN.tt_sec)
    mc = CatBoostRegressor(**CB, random_seed=seed, cat_features=['route_id', 'seg_id']).fit(TRAIN[F], TRAIN.tt_sec)
    for k, X in [('b', TEST), ('s', S), ('w', W)]:
        P['l' + k].append(ml.predict(to_X(X, F))); P['c' + k].append(mc.predict(X[F]))
    print(f'[{PH}] seed {seed} 완료 ({time.time()-t0:.0f}초)', flush=True)
def blend(n, k): return 0.8 * np.mean(P['l' + k][:n], 0) + 0.2 * np.mean(P['c' + k][:n], 0)
out = {}
for n in (1, 5):
    b, s, w = blend(n, 'b'), blend(n, 's'), blend(n, 'w')
    out[f'seed {n}개, 일요일 그대로'] = b
    out[f'seed {n}개, 일요일 반반'] = np.where(sun, 0.5 * s + 0.5 * w, b)
has = 'tt_sec' in test.columns
for name, p in out.items():
    if has:
        a = np.abs(TEST.tt_sec.values - np.round(p, 1))
        print(f'[{PH}] {name:20s} test {a.mean():.3f} | 목금 {a[~sun].mean():.3f} | 일 {a[sun].mean():.3f}', flush=True)
p = out['seed 5개, 일요일 반반']
sub = pd.DataFrame({'row_id': TEST.row_id.values, 'tt_sec': np.round(p, 1)}).set_index('row_id').loc[test.row_id].reset_index()
assert len(sub) == len(test) and sub.tt_sec.notna().all() and not sub.row_id.duplicated().any()
sub.to_csv(f'../submissions/{PH}_E36_seed5_sunmix.csv', index=False); print(f'[{PH}] 저장 ../submissions/{PH}_E36_seed5_sunmix.csv', flush=True)
e35 = pd.read_csv(f'../submissions/{PH}_E35_sunmix.csv').set_index('row_id').tt_sec
s1 = pd.Series(np.round(out['seed 1개, 일요일 반반'], 1), index=TEST.row_id.values)
print(f'[{PH}] 검산: seed 1개 반반 = 기존 E35 파일?', bool((e35 - s1.loc[e35.index]).abs().max() < 1e-9))
