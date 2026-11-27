"""E103: E89 MD 피처 그대로, 평일 3-fold에서 LGB 설정·추가 모델 비교. 모델별 OOF를 experiments/E103_oof/<이름>.parquet 로 저장
실행: python scripts/E103_tune.py <variant> [<variant> ...]
"""
import json, os, re, sys, time, pickle
import numpy as np, pandas as pd, lightgbm as lgb
ROOT = '/Users/kimjuwon/Desktop/에모팀플/bis-travel-time-practice'
os.chdir(ROOT + '/notebooks')
nb = json.load(open('cv_feature_experiments.ipynb')); G = {}
for i in [2, 4, 6, 8, 10, 27, 29, 41]:
    src = ''.join(nb['cells'][i]['source'])
    if i == 2: src = re.sub(r"PHASE = '\w+'", "PHASE = 'final'", src, count=1)
    exec(src, G)
to_X = G['to_X']
cache, meta = pickle.load(open(ROOT + '/experiments/E103_folds.pkl', 'rb'))
FE, LGB, CB = meta['FE'], meta['LGB'], meta['CB']
OUT = ROOT + '/experiments/E103_oof'; os.makedirs(OUT, exist_ok=True)

V = {
    'L_base':   ('fit', 'lgb', {}),
    'W_base':   ('fw', 'lgb', {}),
    'L_nl255':  ('fit', 'lgb', dict(num_leaves=255, min_child_samples=20)),
    'W_nl255':  ('fw', 'lgb', dict(num_leaves=255, min_child_samples=20)),
    'L_slow':   ('fit', 'lgb', dict(learning_rate=0.01, n_estimators=3000)),
    'W_slow':   ('fw', 'lgb', dict(learning_rate=0.01, n_estimators=3000)),
    'L_col5':   ('fit', 'lgb', dict(colsample_bytree=0.5)),
    'W_col5':   ('fw', 'lgb', dict(colsample_bytree=0.5)),
    'L_mc30':   ('fit', 'lgb', dict(min_child_samples=30, reg_lambda=5)),
    'W_mc30':   ('fw', 'lgb', dict(min_child_samples=30, reg_lambda=5)),
    'L_2500':   ('fit', 'lgb', dict(n_estimators=2500)),
    'W_2500':   ('fw', 'lgb', dict(n_estimators=2500)),
    'L_s1':     ('fit', 'lgb', dict(random_state=1)),
    'W_s1':     ('fw', 'lgb', dict(random_state=1)),
    'L_s2':     ('fit', 'lgb', dict(random_state=2)),
    'W_s2':     ('fw', 'lgb', dict(random_state=2)),
    'L_nl511':  ('fit', 'lgb', dict(num_leaves=511, min_child_samples=20)),
    'W_nl511':  ('fw', 'lgb', dict(num_leaves=511, min_child_samples=20)),
    'L_nl1023': ('fit', 'lgb', dict(num_leaves=1023, min_child_samples=20)),
    'W_nl1023': ('fw', 'lgb', dict(num_leaves=1023, min_child_samples=20)),
    'L_nl511d': ('fit', 'lgb', dict(num_leaves=511, min_child_samples=40, n_estimators=2500)),
    'W_nl511d': ('fw', 'lgb', dict(num_leaves=511, min_child_samples=40, n_estimators=2500)),
    'L_nl511s1': ('fit', 'lgb', dict(num_leaves=511, min_child_samples=20, random_state=1)),
    'L_nl511s2': ('fit', 'lgb', dict(num_leaves=511, min_child_samples=20, random_state=2)),
    'L_511mc5':  ('fit', 'lgb', dict(num_leaves=511, min_child_samples=5)),
    'L_511n3k':  ('fit', 'lgb', dict(num_leaves=511, min_child_samples=20, n_estimators=3000)),
    'W_255mc5':  ('fw', 'lgb', dict(num_leaves=255, min_child_samples=5)),
    'W_255n3k':  ('fw', 'lgb', dict(num_leaves=255, min_child_samples=20, n_estimators=3000)),
    'C_d12':     ('fit', 'cb', dict(depth=12, iterations=3000)),
    'L_511mc2':  ('fit', 'lgb', dict(num_leaves=511, min_child_samples=2)),
    'L_1023mc5': ('fit', 'lgb', dict(num_leaves=1023, min_child_samples=5)),
    'W_511mc5':  ('fw', 'lgb', dict(num_leaves=511, min_child_samples=5)),
    'C_base':   ('fit', 'cb', {}),
    'CW_base':  ('fw', 'cb', {}),
    'XW_base':  ('fw', 'xgb', {}),
    'X_base':   ('fit', 'xgb', {}),
}


def run(name, folds):
    src, kind, extra = V[name]
    vkey = 'val' if src == 'fit' else 'vw'
    res = []
    for d in folds:
        c = cache[d]; f, a = c[src], c[vkey]
        t0 = time.time()
        if kind == 'lgb':
            m = lgb.LGBMRegressor(**{**LGB, **extra, 'verbose': -1}).fit(to_X(f, FE), f.tt_sec)
            p = m.predict(to_X(a, FE))
        elif kind == 'cb':
            from catboost import CatBoostRegressor
            m = CatBoostRegressor(**{**CB, **extra}, allow_writing_files=False, cat_features=['route_id', 'seg_id']).fit(f[FE], f.tt_sec)
            p = m.predict(a[FE])
        elif kind == 'xgb':
            import xgboost as xgb
            m = xgb.XGBRegressor(objective='reg:absoluteerror', n_estimators=2000, learning_rate=0.03, max_depth=10, min_child_weight=10,
                                 subsample=0.8, colsample_bytree=0.8, tree_method='hist', enable_categorical=True, max_cat_to_onehot=1, n_jobs=-1, **extra)
            m.fit(to_X(f, FE), f.tt_sec); p = m.predict(to_X(a, FE))
        res.append(pd.DataFrame({'row_id': a.row_id.values, 'date': d, 'tt_sec': a.tt_sec.values if 'tt_sec' in a else np.nan, 'is_turn2': a.is_turn2.values, 'arrive_hour': a.arrive_hour.values, 'route_nm': a.route_nm.astype(str).values, 'pred': p}))
        mae = np.abs(res[-1].tt_sec - p).mean() if d != 'final' else float('nan')
        print(f'{name} {d} {mae:.4f} ({time.time()-t0:.0f}s)', flush=True)
    r = pd.concat(res)
    tag = '_final' if folds == ['final'] else ''
    r.to_parquet(f'{OUT}/{name}{tag}.parquet', index=False)
    if not tag: print(f'== {name} MAE {np.abs(r.tt_sec - r.pred).mean():.4f}', flush=True)


folds = [k for k in cache if k != 'final']
for name in sys.argv[1:]:
    if name.endswith(':final'):
        run(name[:-6], ['final'])
    elif not os.path.exists(f'{OUT}/{name}.parquet'):
        run(name, folds)
