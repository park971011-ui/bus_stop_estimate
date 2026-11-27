"""E67 노선별 일요일 가중치: 일요일 배차 ≠ 토요일 배차 노선만 일요일 행 가중치 x2 모델로 교체
  근거: E66 실습 노선 빼고 검증에서 가중치가 일·토 배차 다른 노선(1, 102)만 개선, 같은 노선(511, 605)은 나빠짐.
        같은 규칙을 1·102에만 적용하면 일요일 27.577 → 27.346 (사후에 찾은 규칙, E45에서 같은 노선 구분이 미리 정한 기준으로 효과 있었음)
  대회: interval_sun != interval_sat 인 노선(2, 704, B1)의 일요일 행만 교체, 나머지(211·311 일요일, 목·금)는 E60 그대로
  모델: E60 조합(LightGBM 0.8 + CatBoost 0.2), 학습 = 대회 train + 실습 train + 실습 test, 일요일 행 가중치 2 (CatBoost도 같은 가중치)
실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E67_sunday_weight_route.py
결과: submissions/final_E67_sunday_weight_route.csv
"""
import os, sys, time
import numpy as np, pandas as pd

ROOT = '/Users/kimjuwon/Desktop/에모팀플/bis-travel-time-practice'
src = open(f'{ROOT}/scripts/E66_sunday_weight.py').read().split('\nt0 = time.time()')[0]
sys.argv = [sys.argv[0], 'cv']
exec(src)

t0 = time.time()
sun = dow(FTE) == 6
diff = (FTE.interval_sun != FTE.interval_sat).values
tgt = sun & diff
print('일·토 배차 다른 노선:', sorted(FTE.route_nm[diff].unique()), f'| 교체할 일요일 행 {tgt.sum():,} / 일요일 {sun.sum():,}', flush=True)
ALL = pd.concat([FTR, PTR, PTE])
p = run_w(ALL, FTE[tgt], 2, 1, cb=True)
e60 = pd.read_csv(f'{ROOT}/submissions/final_E60_sunday_refine.csv').set_index('row_id').tt_sec
new = e60.copy(); ids = FTE.row_id.values[tgt]; new.loc[ids] = np.round(p, 1)
ss = pd.read_csv(f'{ROOT}/data/final/sample_submission.csv')
sub = new.loc[ss.row_id].reset_index(); sub.columns = ['row_id', 'tt_sec']
assert len(sub) == 111563 and (sub.row_id.values == ss.row_id.values).all()
assert sub.tt_sec.notna().all() and not sub.row_id.duplicated().any() and (sub.tt_sec >= 0).all()
keep = FTE.row_id.values[~tgt]
assert (sub.set_index('row_id').tt_sec.loc[keep].values == e60.loc[keep].values).all(), '나머지 = E60'
path = f'{ROOT}/submissions/final_E67_sunday_weight_route.csv'; assert not os.path.exists(path); sub.to_csv(path, index=False)
d = new.loc[ids].values - e60.loc[ids].values
print(f'저장 {path} | 일요일 교체 행 E60 대비 평균 {d.mean():+.2f}초, |차이| {np.abs(d).mean():.2f}초 ({time.time() - t0:.0f}초)', flush=True)
for r in sorted(FTE.route_nm[tgt].unique()):
    m = FTE.route_nm.values[tgt] == r
    print(f'  노선 {r}: 평균 차이 {d[m].mean():+.2f}초, |차이| {np.abs(d[m]).mean():.2f}초', flush=True)
