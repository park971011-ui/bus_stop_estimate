"""E32를 연습 데이터로 다시 학습해서 교수님 채점 코드로 연습 test 채점, 기존 practice_E32.csv와 같은지 확인

실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E32_practice_check.py
결과: 화면 출력만 (파일 안 씀, 약 1.5분)
주의: 노트북 cv_feature_experiments.ipynb 의 셀(2,4,6,8,10,27,29,41번)을 순서대로 실행해서 피처·모델 함수를 가져옴.
      노트북 셀 순서가 바뀌면 이 스크립트도 고쳐야 함.
"""
import json, os, sys, time
import numpy as np, pandas as pd
os.chdir('/Users/kimjuwon/Desktop/에모팀플/bis-travel-time-practice/notebooks')
nb = json.load(open('cv_feature_experiments.ipynb')); G = {}
for i in [2, 4, 6, 8, 10, 27, 29, 41]:
    src = ''.join(nb['cells'][i]['source']); src = __import__('re').sub(r"PHASE = '\w+'", "PHASE = 'practice'", src, count=1) if i == 2 else src; exec(src, G)
TRAIN, TEST, test, PARAMS, feats_of, lgb_fp, cb_fp = (G[k] for k in ['TRAIN','TEST','test','PARAMS','feats_of','lgb_fp','cb_fp'])
G['feats'] = feats_of(['base', 'seg', 'time', 'trip'])
t0 = time.time()
p = 0.8 * lgb_fp(TRAIN, TEST, {**PARAMS, 'min_child_samples': 10, 'learning_rate': 0.02, 'n_estimators': 1500}) + \
    0.2 * cb_fp(TRAIN, TEST, dict(loss_function='MAE', iterations=2000, learning_rate=0.08, depth=10, l2_leaf_reg=3, random_seed=0, verbose=0, thread_count=-1))
sub = pd.DataFrame({'row_id': TEST.row_id.values, 'tt_sec': np.round(p, 1)}).set_index('row_id').loc[test.row_id].reset_index()
sys.path.append('../src'); from evaluate import score
r = score(sub.copy(), test[['row_id', 'tt_sec']], test)
t = test.assign(p=sub.tt_sec.values); D = {3: '목', 4: '금', 6: '일'}
print(f"E32 연습 test MAE {r['MAE']:.3f} | RMSE {r['RMSE']:.3f} | 첨두 {r['첨두 MAE']:.3f} | 비첨두 {r['비첨두 MAE']:.3f}")
print('요일별', {D[pd.Timestamp(d).day_of_week]: round((g.tt_sec - g.p).abs().mean(), 2) for d, g in t.groupby('date')})
print('노선별', {k: round(v, 2) for k, v in r['노선별 MAE'].items()})
old = pd.read_csv('../submissions/practice_E32.csv')
print('기존 제출 파일과 예측 같음:', bool((old.set_index('row_id').tt_sec - sub.set_index('row_id').tt_sec).abs().max() < 1e-9), f'({time.time()-t0:.0f}초)')
