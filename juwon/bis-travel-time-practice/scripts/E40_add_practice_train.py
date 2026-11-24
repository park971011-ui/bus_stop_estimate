"""E40 연습 데이터 train(월·화·수·토, 노선 1·102·511·605)을 대회 학습에 추가. 대회 train 검증 비교 (LightGBM E30 설정만)
  A) 기준 E30: 대회 train만
  B) 대회 train + 연습 train (검증 날짜는 연습 쪽에서도 뺌 → 공정 비교)
연습 test(목·금·일 정답)는 쓰지 않음 (대회 test와 같은 날짜라 규칙 위험).

실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E40_add_practice_train.py
결과: 화면에 검증 점수만 (제출 파일 없음). 약 6분
주의: 노트북 cv_feature_experiments.ipynb 의 셀(2,4,6,8,10,27,29,41번)을 실행해서 피처·모델 함수를 가져옴.
"""
import json, os, re, time
import numpy as np, pandas as pd, lightgbm as lgb
os.chdir('/Users/kimjuwon/Desktop/에모팀플/bis-travel-time-practice/notebooks')
nb = json.load(open('cv_feature_experiments.ipynb')); G = {}
for i in [2, 4, 6, 8, 10, 27, 29, 41]:
    src = ''.join(nb['cells'][i]['source'])
    if i == 2: src = re.sub(r"PHASE = '\w+'", "PHASE = 'final'", src, count=1)
    exec(src, G)
TR = G['TRAIN']
# 연습 train을 같은 함수로 피처 만들기 (last_seq는 입력값만 쓰므로 연습 train+test 입력으로 계산)
ptr = pd.read_parquet('../data/practice/train.parquet'); pte = pd.read_parquet('../data/practice/test.parquet').drop(columns='tt_sec')
G['train'], G['test'] = ptr, pte
PTR = G['add_busy'](G['add_trip'](G['add_static'](ptr)))
PTR.index = PTR.index + 10_000_000
ALL = pd.concat([TR, PTR])
for c in G['CATS']:
    G['CAT_DTYPE'][c] = pd.CategoricalDtype(sorted(pd.concat([ALL[c], G['TEST'][c]]).unique()))
LGB = {**G['PARAMS'], 'min_child_samples': 10, 'learning_rate': 0.02, 'n_estimators': 1500}
E5 = G['feats_of'](['base', 'seg', 'time', 'trip'])
DATES, DAY, summarize, mae, to_X = G['DATES'], G['DAY'], G['summarize'], G['mae'], G['to_X']

def cv(name, extra):
    t0 = time.time(); sc = {}
    for d in DATES:
        f = TR[TR.date != d]
        if extra: f = pd.concat([f, PTR[PTR.date != d]])
        v = TR[TR.date == d]
        sc[DAY[pd.Timestamp(d).day_of_week]] = mae(v.tt_sec, lgb.LGBMRegressor(**LGB).fit(to_X(f, E5), f.tt_sec).predict(to_X(v, E5)))
    r = summarize(sc)
    print(f"{name:26s} 예상 {r['예상']:.3f} | 평일 {r['평일']:.3f} | 토 {r['토']:.3f} | "
          + ' '.join(f'{k} {sc[k]:.3f}' for k in sc) + f'  ({time.time() - t0:.0f}초)', flush=True)
    return r

a = cv('A 대회 train만', False)
b = cv('B + 연습 train', True)
print(f"B - A = {b['예상'] - a['예상']:+.3f}")
