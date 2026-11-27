"""E75 E74 피처 조합을 연습 데이터로 확인 (진짜 처음 보는 목·금)
  연습 train(월·화·수·토)으로 학습 → 연습 test(목·금·일, 정답 있음) 채점. 대회와 같은 구조.
  비교: E41 피처 35개 vs E74 조합 43개 (+ peak2, wkhour, turn, turn_est). LightGBM E41 설정.
  채택 기준(미리 고정): 연습 목·금 MAE가 좋아질 것 (일요일은 따로 일요일 모델을 쓰므로 참고만)
실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E75_combo_practice_check.py
"""
import sys, time
import numpy as np, pandas as pd, lightgbm as lgb

ROOT = '/Users/kimjuwon/Desktop/에모팀플/bis-travel-time-practice'
src = open(f'{ROOT}/scripts/E65_gap_minute.py').read().split('\ndef add_gap')[0]
sys.argv = [sys.argv[0], 'practice']
exec(src)                                    # PH = practice: TRAIN, TEST, TR(회차 힌트), F, LGB, G
assert G['PHASE'] == 'practice' and 'tt_sec' in TEST.columns
C_EST = 3574.0


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


turn = turn_of(TRAIN)
f, v = add_turn_hint(TRAIN, turn), add_turn_hint(TEST, turn)
mf = np.full(len(f), np.nan)
for dd in f.date.unique():
    m = (f.date == dd).values; mf[m] = seg_med(f[~m], f[m])
f, v = add_est(f, mf), add_est(v, seg_med(f, v))
FG = G['FEATS']
COMBO = F + FG['peak2'] + FG['wkhour'] + FG['turn'] + ['turn_est']
y = v.tt_sec.values; dw = pd.to_datetime(v.date).dt.day_of_week.values
t0 = time.time()
out = {}
for name, feats in [('E41 피처 35개', F), ('E74 조합 43개', COMBO)]:
    P = [lgb.LGBMRegressor(**{**LGB, 'random_state': s}).fit(to_X(f, feats), f.tt_sec).predict(to_X(v, feats)) for s in range(2)]
    p = np.mean(P, axis=0); e = np.abs(y - p); out[name] = e
    print(f'{name}: 목 {e[dw == 3].mean():.3f} 금 {e[dw == 4].mean():.3f} | 목·금 {e[dw < 5].mean():.3f} | 일(참고) {e[dw == 6].mean():.3f} | 전체 {e.mean():.3f} ({time.time() - t0:.0f}초)', flush=True)
a, b = out['E41 피처 35개'], out['E74 조합 43개']
rn = v.route_nm.values
print('노선별 목·금 차이:', {r: round(b[(rn == r) & (dw < 5)].mean() - a[(rn == r) & (dw < 5)].mean(), 3) for r in sorted(set(rn))}, flush=True)
print(f'→ 목·금 {b[dw < 5].mean() - a[dw < 5].mean():+.3f} : {"채택" if b[dw < 5].mean() < a[dw < 5].mean() else "채택 안 함"}', flush=True)
