"""E39 대회 train 검증 비교 (LightGBM E30 설정만):
  A) 기준 E30 (E5 피처 25개)          — E38A와 같음
  B) + n_bus_hour (시간대 버스 수) + 배차 비율(interval / interval_wd)
  C) 예측 대상 log(tt)  → exp로 되돌림
  D) 예측 대상 tt / dist (1m당 초) → × dist로 되돌림

실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E39_target_busy.py final   (또는 practice)
결과: 화면에 검증 점수만 (제출 파일 없음). 약 15분
주의: 노트북 cv_feature_experiments.ipynb 의 셀(2,4,6,8,10,27,29,41번)을 실행해서 피처·모델 함수를 가져옴.
"""
import json, os, sys, time
import numpy as np, pandas as pd, lightgbm as lgb
os.chdir('/Users/kimjuwon/Desktop/에모팀플/bis-travel-time-practice/notebooks')
PH = sys.argv[1]
nb = json.load(open('cv_feature_experiments.ipynb')); G = {}
for i in [2, 4, 6, 8, 10, 27, 29, 41]:
    src = ''.join(nb['cells'][i]['source'])
    if i == 2: src = __import__('re').sub(r"PHASE = '\w+'", f"PHASE = '{PH}'", src, count=1)
    exec(src, G)
TR = G['TRAIN'].copy()
TR['int_ratio'] = TR.interval / TR.interval_wd
LGB = {**G['PARAMS'], 'min_child_samples': 10, 'learning_rate': 0.02, 'n_estimators': 1500}
E5 = G['feats_of'](['base', 'seg', 'time', 'trip'])
DATES, DAY, summarize, mae, to_X = G['DATES'], G['DAY'], G['summarize'], G['mae'], G['to_X']
D = np.maximum(TR.dist_m, 1)

def cv(name, feats, fwd=lambda y, d: y, back=lambda p, d: p, w=False):
    t0 = time.time(); sc = {}
    for d in DATES:
        f, v = TR[TR.date != d], TR[TR.date == d]
        m = lgb.LGBMRegressor(**LGB).fit(to_X(f, feats), fwd(f.tt_sec, D[f.index]), sample_weight=D[f.index] if w else None)
        sc[DAY[pd.Timestamp(d).day_of_week]] = mae(v.tt_sec, back(m.predict(to_X(v, feats)), D[v.index].values))
    r = summarize(sc)
    print(f"[{PH}] {name:22s} 예상 {r['예상']:.3f} | 평일 {r['평일']:.3f} | 토 {r['토']:.3f} | "
          + ' '.join(f'{k} {sc[k]:.3f}' for k in sc) + f'  ({time.time() - t0:.0f}초)', flush=True)
    return r

out = {}
out['A'] = cv('A 기준 E30', E5)
out['B'] = cv('B +버스 수·배차 비율', E5 + ['n_bus_hour', 'int_ratio'])
out['C'] = cv('C log(tt) 예측', E5, lambda y, d: np.log1p(y), lambda p, d: np.expm1(p))
out['D'] = cv('D 1m당 초 예측', E5, lambda y, d: y / d, lambda p, d: p * d, w=True)   # 거리 가중치: |tt/거리 - p|×거리 = |tt - p×거리|
print(f'[{PH}] 기준 대비:', {k: round(out[k]['예상'] - out['A']['예상'], 3) for k in 'BCD'})
