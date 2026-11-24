"""E32 대회 제출 파일 + 대회 train 4-fold 검증 (E0 평균 예측, LightGBM, CatBoost, E32, 섞는 비율)

실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E32_final_cv.py
결과: submissions/final_E32.csv, experiments/oof_final_E32.parquet (약 7분)
주의: 노트북 cv_feature_experiments.ipynb 의 셀(2,4,6,8,10,27,29,41번)을 순서대로 실행해서 피처·모델 함수를 가져옴.
      노트북 셀 순서가 바뀌면 이 스크립트도 고쳐야 함.
"""
import json, os, time
import numpy as np, pandas as pd
os.chdir('/Users/kimjuwon/Desktop/에모팀플/bis-travel-time-practice/notebooks')
nb = json.load(open('cv_feature_experiments.ipynb')); G = {}
for i in [2, 4, 6, 8, 10, 27, 29, 41]:
    src = ''.join(nb['cells'][i]['source'])
    if i == 2: src = __import__('re').sub(r"PHASE = '\w+'", "PHASE = 'final'", src, count=1); assert "PHASE = 'final'" in src
    exec(src, G)
TRAIN, TEST, DATES, DAY, cv_of, oof_of, lgb_fp, cb_fp, PARAMS, feats_of, TURN_SEQ = (G[k] for k in
    ['TRAIN','TEST','DATES','DAY','cv_of','oof_of','lgb_fp','cb_fp','PARAMS','feats_of','TURN_SEQ'])
print('PHASE', G['PHASE'], '| TRAIN', TRAIN.shape, '| TEST', TEST.shape)
print('회차 구간(노선별 seq):', TURN_SEQ.to_dict(), flush=True)
FEATS = feats_of(['base', 'seg', 'time', 'trip']); G['feats'] = FEATS
LGB = {**PARAMS, 'min_child_samples': 10, 'learning_rate': 0.02, 'n_estimators': 1500}
CB = dict(loss_function='MAE', iterations=2000, learning_rate=0.08, depth=10, l2_leaf_reg=3, random_seed=0, verbose=0, thread_count=-1)

# 1) 제출 파일 (노트북 14번과 같은 방식)
t0 = time.time()
p = 0.8 * lgb_fp(TRAIN, TEST, LGB) + 0.2 * cb_fp(TRAIN, TEST, CB)
test = G['test']
sub = pd.DataFrame({'row_id': TEST.row_id.values, 'tt_sec': np.round(p, 1)}).set_index('row_id').loc[test.row_id].reset_index()
ss = pd.read_csv('../data/final/sample_submission.csv')
assert len(sub) == len(ss) and set(sub.row_id) == set(ss.row_id) and sub.tt_sec.notna().all() and not sub.row_id.duplicated().any()
sub.to_csv('../submissions/final_E32.csv', index=False)
print(f'저장 ../submissions/final_E32.csv {sub.shape} | 결측 0 중복 0 | 예측 평균 {sub.tt_sec.mean():.1f} ({time.time()-t0:.0f}초)', flush=True)

# 2) E0 평균 예측 (노선·방향·정류장순번·시간대 평균, 없으면 노선·방향·순번)
k3 = ['route_id', 'direction', 'stop_seq', 'arrive_hour']; k2 = ['route_id', 'direction', 'stop_seq']
e0 = pd.Series(np.nan, index=TRAIN.index)
for d in DATES:
    fit, val = TRAIN[TRAIN.date != d], TRAIN[TRAIN.date == d]
    v = val.join(fit.groupby(k3).tt_sec.mean().rename('a'), on=k3).join(fit.groupby(k2).tt_sec.mean().rename('b'), on=k2)
    e0[val.index] = v.a.fillna(v.b).fillna(fit.tt_sec.mean()).values
r0 = cv_of(e0); print('E0 평균 예측', {k: round(v, 3) for k, v in r0.items()}, flush=True)

# 3) E32 4-fold
o_lgb = oof_of(lgb_fp, LGB); print('LightGBM', {k: round(v, 3) for k, v in cv_of(o_lgb).items()}, flush=True)
o_cb = oof_of(cb_fp, CB); print('CatBoost', {k: round(v, 3) for k, v in cv_of(o_cb).items()}, flush=True)
o = 0.8 * o_lgb + 0.2 * o_cb; r = cv_of(o); print('E32', {k: round(v, 3) for k, v in r.items()}, flush=True)
pd.DataFrame({'row_id': TRAIN.row_id.values, 'lgb': o_lgb.values, 'cb': o_cb.values, 'e32': o.values, 'e0': e0.values}).to_parquet('../experiments/oof_final_E32.parquet')
for w in np.round(np.arange(0.5, 1.01, 0.1), 1):
    print('  LightGBM 비율', w, round(cv_of(w * o_lgb + (1 - w) * o_cb)['예상'], 3))
tt = TRAIN.assign(p=o.values, e0=e0.values)
print('노선별 검증 MAE (E32 / E0):', {k: (round((g.tt_sec - g.p).abs().mean(), 2), round((g.tt_sec - g.e0).abs().mean(), 2)) for k, g in tt.groupby('route_nm')})
print(f'끝 ({time.time()-t0:.0f}초)')
