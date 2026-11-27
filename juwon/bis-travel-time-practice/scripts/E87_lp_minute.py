"""E87 분 복원 + 회차 대기 역산 (LP, scripts/lp_dwell.py)
  발견: arrive_hour는 구간 '출발' 시각의 시. 진짜 tt로 검사하면 출발 해석은 사슬 3381개 중 모순 4개, 도착 해석은 387개.
        → 차량 하루를 시각순으로 놓고 평소 시간을 이어 붙이면, 모든 행이 자기 시 안에 들어야 해서 분까지 거의 정해짐
        (평소 시간으로 푼 분 vs 진짜 tt로 푼 분: 평균 2.3분 차이, 그냥 30분이라 하면 15.1분)
  피처: lp_min(출발 분 추정), lp_tod(시 + 분/60), lp_D(회차 행 대기 역산, 나머지 결측)
  평소 시간 = 구간 중앙값 (학습 행은 자기 날짜 빼고, 검증 행은 학습 날짜로). 입력만 사용.
  기준 = E79 피처 45개. 검증 1: 대회 train 4-fold / 검증 2: 연습 train → 연습 test 진짜 목·금
  채택 기준(미리 고정): 두 검증 모두 기준보다 좋아짐 (대회는 평일 2/3일 이상)
실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E87_lp_minute.py final   (또는 practice)
"""
import sys, time
import numpy as np, pandas as pd, lightgbm as lgb
ROOT = '/Users/kimjuwon/Desktop/에모팀플/bis-travel-time-practice'
exec(open(f'{ROOT}/scripts/E78_hour_cross.py').read().split('\nFG = G')[0])
sys.path.insert(0, f'{ROOT}/scripts')
from lp_dwell import lp_dwell
FG = G['FEATS']
BASE = F + FG['peak2'] + FG['wkhour'] + FG['turn'] + ['turn_est'] + XC
LPM, LPD = ['lp_min', 'lp_tod'], ['lp_D']


def add_lp(df, med):
    r = lp_dwell(df, med, 0.3, 0.05)
    d = df.copy(); d['lp_min'] = r.lp_min.values; d['lp_tod'] = d.arrive_hour + d.lp_min / 60; d['lp_D'] = r.lp_D.values
    return d


def prep_pair(f, v):
    f, v = with_est(f, v)
    mf = np.full(len(f), np.nan)
    for dd in f.date.unique():
        m = (f.date == dd).values; mf[m] = seg_med(f[~m], f[m])
    return add_lp(f, mf), add_lp(v, seg_med(f, v))


SETS = [('기준 (E79 45개)', BASE), ('+ 분 복원', BASE + LPM), ('+ 회차 역산', BASE + LPD), ('+ 둘 다', BASE + LPM + LPD)]
t0 = time.time()
if PHASE_ARG == 'final':
    TA = add_cross(TR)
    folds = [prep_pair(TA[(TA.date != dd).values], TA[(TA.date == dd).values]) for dd in DATES]
    print(f'fold 준비 {time.time() - t0:.0f}초', flush=True)
    for name, feats in SETS:
        o = pd.Series(np.nan, index=TA.index)
        for f, v in folds:
            o[v.index] = lgb.LGBMRegressor(**LGB).fit(to_X(f, feats), f.tt_sec).predict(to_X(v, feats))
        r = cv_of(o); e = np.abs(TA.tt_sec - o); tn = (TA.is_turn2 == 1).values if 'is_turn2' in TA else np.zeros(len(TA), bool)
        print(f"[final] {name:14s} 예상 {r['예상']:.3f} | 월 {r['월']:.3f} 화 {r['화']:.3f} 수 {r['수']:.3f} 토 {r['토']:.3f} ({time.time() - t0:.0f}초)", flush=True)
else:
    turn = turn_of(TRAIN)
    f, v = prep_pair(add_cross(add_turn_hint(TRAIN, turn)), add_cross(add_turn_hint(TEST, turn)))
    y = v.tt_sec.values; dw = pd.to_datetime(v.date).dt.day_of_week.values; tn = (v.is_turn2 == 1).values
    for name, feats in SETS:
        P = [lgb.LGBMRegressor(**{**LGB, 'random_state': s}).fit(to_X(f, feats), f.tt_sec).predict(to_X(v, feats)) for s in range(2)]
        e = np.abs(y - np.mean(P, axis=0))
        print(f'[practice] {name:14s} 목 {e[dw == 3].mean():.3f} 금 {e[dw == 4].mean():.3f} | 목·금 {e[dw < 5].mean():.3f} | 일(참고) {e[dw == 6].mean():.3f} | 회차 행 {e[tn].mean():.1f} ({time.time() - t0:.0f}초)', flush=True)
