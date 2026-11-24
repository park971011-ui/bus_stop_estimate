"""E46 = 목·금은 E44(E41 0.5 + 평일 전용 LightGBM 0.5), 일요일은 E45(노선별 일요일 섞기). 두 파일을 합치기만 함.
  E44는 목·금만, E45는 일요일만 E41과 다르므로 서로 겹치지 않음.
실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E46_combine_E44_E45.py practice   (또는 final)
결과: submissions/{practice|final}_E46_wd_route_sunmix.csv (E44·E45 파일이 먼저 있어야 함)
"""
import sys, os
import numpy as np, pandas as pd
os.chdir('/Users/kimjuwon/Desktop/에모팀플/bis-travel-time-practice')
PH = sys.argv[1]
te = pd.read_parquet(f'data/{PH}/test.parquet').set_index('row_id')
ss = pd.read_csv(f'data/{PH}/sample_submission.csv') if os.path.exists(f'data/{PH}/sample_submission.csv') else None
r = lambda n: pd.read_csv(f'submissions/{PH}_{n}.csv').set_index('row_id').tt_sec
e41, e44, e45 = r('E41_turnhint_sunmix'), r('E44_weekday_lgb'), r('E45_route_sunmix')
ids = e41.index
sun = pd.to_datetime(te.loc[ids].date).dt.day_of_week.values == 6
assert (e44.loc[ids][sun] == e41[sun]).all() and (e45.loc[ids][~sun] == e41[~sun]).all(), 'E44는 목·금만, E45는 일요일만 달라야 함'
sub = pd.DataFrame({'row_id': ids, 'tt_sec': np.where(sun, e45.loc[ids].values, e44.loc[ids].values)})
if ss is not None:
    assert (sub.row_id.values == ss.row_id.values).all()
assert sub.tt_sec.notna().all() and not sub.row_id.duplicated().any()
print(f'[{PH}] 음수 예측 {(sub.tt_sec < 0).sum()}개')
assert PH != 'final' or (sub.tt_sec >= 0).all(), '대회 파일에 음수'
path = f'submissions/{PH}_E46_wd_route_sunmix.csv'; sub.to_csv(path, index=False)
print(f'[{PH}] 저장 {path} {sub.shape} | 목·금 = E44, 일 = E45 | 결측·중복 0')
if 'tt_sec' in te.columns:
    y = te.loc[ids].tt_sec.values
    for n, p in [('E41', e41.values), ('E44', e44.loc[ids].values), ('E45', e45.loc[ids].values), ('E46', sub.tt_sec.values)]:
        a = np.abs(y - p); print(f'[{PH}] {n} 연습 test {a.mean():.3f} | 목금 {a[~sun].mean():.3f} | 일 {a[sun].mean():.3f}')
