"""E83 길이 편향 넓히기 + 운행 규모 (기준 = E79 피처 45개: E74 조합 + 정각 넘김)
  A trip: 운행이 시를 몇 번 넘겼나(trip_span = 운행 마지막 행 시 − 첫 행 시), 운행 구간 평소 시간 합(trip_usual), 구간 수(trip_n),
          trip_lb = trip_span×3600 / trip_usual (운행이 최소 이만큼은 걸렸다 ÷ 평소)
  B win: 앞뒤 k개 구간(k = 3, 8) 안 정각 넘김 횟수(win{k}_cross)와 그 구간들 평소 시간 합(win{k}_usual)
  C svc: 그날 노선 차량 수(svc_nveh), 이 차량 그날 운행 수(svc_ntrip), 같은 날·노선·시간 차량 수(svc_nveh_h) — 각각 다른 날 평균 대비 비율도
  평소 시간 = 구간 중앙값 (학습 행은 자기 날짜 빼고, 검증 행은 학습 날짜로)
  검증 1: 대회 train 4-fold / 검증 2: 연습 train → 연습 test 진짜 목·금
  채택 기준(미리 고정): 두 검증 모두 기준보다 좋아짐 (대회는 평일 2/3일 이상)
실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E83_more_bias.py final   (또는 practice)
"""
import sys, time
import numpy as np, pandas as pd, lightgbm as lgb

ROOT = '/Users/kimjuwon/Desktop/에모팀플/bis-travel-time-practice'
exec(open(f'{ROOT}/scripts/E78_hour_cross.py').read().split('\nFG = G')[0])   # PHASE_ARG, add_cross, with_est, seg_med, G, F, LGB ...
FG = G['FEATS']
BASE = F + FG['peak2'] + FG['wkhour'] + FG['turn'] + ['turn_est'] + XC
TK = ['date', 'vehicle_id', 'route_id', 'trip_no']
A = ['trip_span', 'trip_usual', 'trip_n', 'trip_lb']
B = ['win3_cross', 'win3_usual', 'win8_cross', 'win8_usual']
C = ['svc_nveh', 'svc_ntrip', 'svc_nveh_h', 'svc_nveh_r', 'svc_nveh_h_r']


def add_new(df, med):
    d = df.copy(); d['_m'] = med; d['_m'] = d._m.fillna(np.nanmedian(med))
    s = d.sort_values(TK + ['stop_seq'])
    g = s.groupby(TK)
    s['trip_span'] = g.arrive_hour.transform('max') - g.arrive_hour.transform('min')
    s['trip_usual'] = g._m.transform('sum'); s['trip_n'] = g._m.transform('size')
    s['trip_lb'] = s.trip_span * 3600 / s.trip_usual
    cr = (s.h_cross.fillna(0) > 0).astype(float)
    for k in [3, 8]:
        s[f'win{k}_cross'] = cr.groupby([s[c] for c in TK]).transform(lambda x: x.rolling(2 * k + 1, center=True, min_periods=1).sum())
        s[f'win{k}_usual'] = s.groupby(TK)._m.transform(lambda x: x.rolling(2 * k + 1, center=True, min_periods=1).sum())
    s['svc_nveh'] = s.groupby(['date', 'route_id']).vehicle_id.transform('nunique')
    s['svc_ntrip'] = s.groupby(['date', 'vehicle_id']).trip_no.transform('nunique')
    s['svc_nveh_h'] = s.groupby(['date', 'route_id', 'arrive_hour']).vehicle_id.transform('nunique')
    # 다른 날 평균 대비 (입력만 사용, 같은 요일 유형 구분 없이)
    day = s.groupby(['date', 'route_id']).vehicle_id.nunique().rename('n').reset_index()
    s = s.join(day.groupby('route_id').n.mean().rename('_nv'), on='route_id').set_index(s.index)
    dh = s.groupby(['date', 'route_id', 'arrive_hour']).vehicle_id.nunique().rename('n').reset_index()
    s = s.join(dh.groupby(['route_id', 'arrive_hour']).n.mean().rename('_nvh'), on=['route_id', 'arrive_hour']).set_index(s.index)
    s['svc_nveh_r'] = s.svc_nveh / s._nv; s['svc_nveh_h_r'] = s.svc_nveh_h / s._nvh
    return s.drop(columns=['_m', '_nv', '_nvh']).loc[df.index]


def prep_pair(f, v):
    f, v = with_est(f, v)
    mf = np.full(len(f), np.nan)
    for dd in f.date.unique():
        m = (f.date == dd).values; mf[m] = seg_med(f[~m], f[m])
    return add_new(f, mf), add_new(v, seg_med(f, v))


SETS = [('기준 (E79 45개)', BASE), ('+ A 운행 길이 편향', BASE + A), ('+ B 주변 정각 넘김', BASE + B), ('+ C 운행 규모', BASE + C), ('+ A B C 전부', BASE + A + B + C)]
t0 = time.time()
if PHASE_ARG == 'final':
    TA = add_cross(TR)
    folds = [prep_pair(TA[(TA.date != dd).values], TA[(TA.date == dd).values]) for dd in DATES]
    print(f'fold 준비 {time.time() - t0:.0f}초', flush=True)
    for name, feats in SETS:
        o = pd.Series(np.nan, index=TA.index)
        for f, v in folds:
            o[v.index] = lgb.LGBMRegressor(**LGB).fit(to_X(f, feats), f.tt_sec).predict(to_X(v, feats))
        r = cv_of(o)
        print(f"[final] {name:18s} 예상 {r['예상']:.3f} | 월 {r['월']:.3f} 화 {r['화']:.3f} 수 {r['수']:.3f} 토 {r['토']:.3f} ({time.time() - t0:.0f}초)", flush=True)
else:
    turn = turn_of(TRAIN)
    f, v = prep_pair(add_cross(add_turn_hint(TRAIN, turn)), add_cross(add_turn_hint(TEST, turn)))
    y = v.tt_sec.values; dw = pd.to_datetime(v.date).dt.day_of_week.values
    for name, feats in SETS:
        P = [lgb.LGBMRegressor(**{**LGB, 'random_state': s}).fit(to_X(f, feats), f.tt_sec).predict(to_X(v, feats)) for s in range(2)]
        e = np.abs(y - np.mean(P, axis=0))
        print(f'[practice] {name:18s} 목 {e[dw == 3].mean():.3f} 금 {e[dw == 4].mean():.3f} | 목·금 {e[dw < 5].mean():.3f} | 일(참고) {e[dw == 6].mean():.3f} ({time.time() - t0:.0f}초)', flush=True)
