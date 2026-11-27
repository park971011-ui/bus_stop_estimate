"""E78 정각 넘김 피처 (길이 편향)
  가설: 구간이 길수록 그 사이에 정각(hh:00)을 지날 확률이 높음 (확률 ≈ 통행시간/3600).
        → "이 구간에서 정각을 넘겼다"는 사실 자체가 평소보다 길었다는 증거. 입력(arrive_hour)만으로 앎.
  확인: 정각 넘긴 행(3%) 실제 중앙값 136초인데 E41 예측 중앙값 120초, 잔차 평균 +19.8초 (안 넘긴 행 +2.5초)
  피처: h_cross = 같은 운행 바로 다음 행(stop_seq+1)의 시 − 이 행의 시 (다음 행 없으면 결측)
        h_cross_prev = 이 행의 시 − 바로 앞 행(stop_seq−1)의 시
  검증 1: 대회 train 날짜별 4-fold (E41 피처 기준, E74 조합 기준)
  검증 2: 연습 train 학습 → 연습 test 진짜 목·금 (E41 피처 기준, E74 조합 기준)
  채택 기준(미리 고정): 검증 1 예상 점수 개선 + 평일 2/3일 이상 개선, 그리고 검증 2 목·금 개선
실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E78_hour_cross.py final   (또는 practice)
"""
import sys, time
import numpy as np, pandas as pd, lightgbm as lgb

ROOT = '/Users/kimjuwon/Desktop/에모팀플/bis-travel-time-practice'
PHASE_ARG = sys.argv[1]
src = open(f'{ROOT}/scripts/E65_gap_minute.py').read().split('\ndef add_gap')[0]
sys.argv = [sys.argv[0], PHASE_ARG]
exec(src)
C_EST = 3574.0
XC = ['h_cross', 'h_cross_prev']


def add_cross(df):
    s = df.sort_values(['date', 'vehicle_id', 'route_id', 'trip_no', 'stop_seq'])
    g = s.groupby(['date', 'vehicle_id', 'route_id', 'trip_no'])
    ns, nh, ps, ph = g.stop_seq.shift(-1), g.arrive_hour.shift(-1), g.stop_seq.shift(1), g.arrive_hour.shift(1)
    s = s.copy()
    s['h_cross'] = np.where(ns == s.stop_seq + 1, nh - s.arrive_hour, np.nan)
    s['h_cross_prev'] = np.where(ps == s.stop_seq - 1, s.arrive_hour - ph, np.nan)
    return s.loc[df.index]


def seg_med(src_, apply):
    md = src_.groupby(SEGK).tt_sec.median()
    return apply[SEGK].join(md.rename('x'), on=SEGK).x.values


def add_est(df, med):
    df = df.copy(); df['_m'] = med
    hs = df.groupby(V + ['arrive_hour'])._m.sum()
    k = lambda s: pd.MultiIndex.from_arrays([df.date, df.vehicle_id, df.arrive_hour + s])
    inner = k(-1).isin(hs.index) & k(1).isin(hs.index)
    others = hs.reindex(k(0)).values - np.nan_to_num(df._m.values)
    df['turn_est'] = np.where((df.is_turn2 == 1) & inner, C_EST - others, np.nan)
    return df.drop(columns='_m')


def with_est(f, v):
    mf = np.full(len(f), np.nan)
    for dd in f.date.unique():
        m = (f.date == dd).values; mf[m] = seg_med(f[~m], f[m])
    return add_est(f, mf), add_est(v, seg_med(f, v))


FG = G['FEATS']
COMBO = F + FG['peak2'] + FG['wkhour'] + FG['turn'] + ['turn_est']
SETS = [('E41 피처', F), ('E41 + 정각 넘김', F + XC), ('E74 조합', COMBO), ('E74 조합 + 정각 넘김', COMBO + XC)]
t0 = time.time()

if PHASE_ARG == 'final':
    TA = add_cross(TR)
    folds = []
    for dd in DATES:
        fm = (TA.date != dd).values
        folds.append(with_est(TA[fm], TA[~fm]))
    rows = []
    for name, feats in SETS:
        o = pd.Series(np.nan, index=TA.index)
        for f, v in folds:
            o[v.index] = lgb.LGBMRegressor(**LGB).fit(to_X(f, feats), f.tt_sec).predict(to_X(v, feats))
        r = cv_of(o)
        e = np.abs(TA.tt_sec - o); cr = (TA.h_cross >= 1).values
        rows.append({'실험': name, '월': r['월'], '화': r['화'], '수': r['수'], '토': r['토'], '예상': r['예상'],
                     '정각 넘긴 행': e[cr].mean(), '나머지': e[~cr].mean()})
        print(f"[final] {name:18s} 예상 {r['예상']:.3f} | 월 {r['월']:.3f} 화 {r['화']:.3f} 수 {r['수']:.3f} 토 {r['토']:.3f} | 정각 넘긴 행 {e[cr].mean():.1f} 나머지 {e[~cr].mean():.3f} ({time.time() - t0:.0f}초)", flush=True)
    pd.DataFrame(rows).set_index('실험').round(3).to_csv(f'{ROOT}/experiments/E78_cv_final.csv')
else:
    turn = turn_of(TRAIN)
    f, v = with_est(add_cross(add_turn_hint(TRAIN, turn)), add_cross(add_turn_hint(TEST, turn)))
    y = v.tt_sec.values; dw = pd.to_datetime(v.date).dt.day_of_week.values; cr = (v.h_cross >= 1).values
    for name, feats in SETS:
        P = [lgb.LGBMRegressor(**{**LGB, 'random_state': s}).fit(to_X(f, feats), f.tt_sec).predict(to_X(v, feats)) for s in range(2)]
        e = np.abs(y - np.mean(P, axis=0))
        print(f'[practice] {name:18s} 목 {e[dw == 3].mean():.3f} 금 {e[dw == 4].mean():.3f} | 목·금 {e[dw < 5].mean():.3f} | 일(참고) {e[dw == 6].mean():.3f} | 정각 넘긴 행 {e[cr].mean():.1f} ({time.time() - t0:.0f}초)', flush=True)
