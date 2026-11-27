"""E86 회차 행 오차 분석 (평일 OOF = E82와 같은 E79 방식)
  회차 행마다 피처 + OOF 잔차를 모아 parquet으로 저장 → 무엇이 잔차를 설명하는지 확인
실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E86_turn_analysis.py final
"""
import sys, time
import numpy as np, pandas as pd
ROOT = '/Users/kimjuwon/Desktop/에모팀플/bis-travel-time-practice'
exec(open(f'{ROOT}/scripts/E79_cross_wd.py').read().split('\nt0 = time.time()')[0])
O = pd.read_parquet(f'{ROOT}/experiments/E82_oof.parquet')[['row_id', 'pB', 'pW']]
WD = dow(TRAIN) < 5
out = []
for d in sorted(TRAIN[WD].date.unique()):
    m = (TRAIN.date == d).values
    f, v = prep(TRAIN[~m], TRAIN[m])
    out.append(v)
A = pd.concat(out).merge(O, on='row_id')
A['p'] = 0.5 * A.pB + 0.5 * A.pW
A.to_parquet(f'{ROOT}/experiments/E86_wd_rows.parquet')
print('저장', len(A), (A.is_turn2 == 1).sum())
