"""E72 피처 전부 넣기 (대회 train 날짜별 4-fold, E41 LightGBM 25.227과 비교)
  ALL = 노트북 피처 묶음 전부(base, seg, time, te, trip, busy, station, peak2, wkhour, turn, struct, share, center, geo, tolast)
        + 회차 힌트 10개(E41) + 추정 회차 시간(E56) + 기록 빠짐(E65A) + 분 단위 위치(E65B) + 같은 날 연습 기록(E69 A+B)
  ALL−px = ALL에서 같은 날 연습 기록(E69)만 뺌
  뺀 것: vehicle_id (그냥 번호, E24에서 크게 나빠짐)
  정답을 쓰는 피처(te, turn_est, est_min)는 fold마다 학습 날짜로만, 학습 행은 자기 날짜를 빼고 계산
실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E72_all_features.py
"""
import sys, time
import numpy as np, pandas as pd, lightgbm as lgb

ROOT = '/Users/kimjuwon/Desktop/에모팀플/bis-travel-time-practice'
sys.argv = [sys.argv[0], 'cv']
exec(open(f'{ROOT}/scripts/E69_practice_sameday.py').read().split('\nt0 = time.time()')[0])   # E65 앞부분 + add_px
e65 = open(f'{ROOT}/scripts/E65_gap_minute.py').read()
exec('def add_gap' + e65.split('\ndef add_gap')[1].split('\nTR = add_gap(TR)')[0])                 # add_gap, add_minute, seg_med

C_EST = 3574.0


def add_est(df, med):
    df = df.copy(); df['_m'] = med
    hs = df.groupby(V + ['arrive_hour'])._m.sum()
    k = lambda s: pd.MultiIndex.from_arrays([df.date, df.vehicle_id, df.arrive_hour + s])
    inner = k(-1).isin(hs.index) & k(1).isin(hs.index)
    others = hs.reindex(k(0)).values - np.nan_to_num(df._m.values)
    df['turn_est'] = np.where((df.is_turn2 == 1) & inner, C_EST - others, np.nan)
    return df.drop(columns='_m')


GROUPS = [g for g in G['FEATS'] if g != 'te']
BASEF = sum((G['FEATS'][g] for g in GROUPS), []) + H
TE = G['FEATS']['te']
EXTRA = ['turn_est', 'gap_before', 'gap_after', 'est_min', 'est_time']
PX = ['px_seg_hr', 'px_seg_r', 'px_seg_n', 'px_near_r', 'px_near_n', 'px_glob_r']

t0 = time.time()
TA = add_px(add_gap(TR))
print(f'피처 묶음 {len(GROUPS) + 1}개 | ALL 피처 {len(BASEF + TE + EXTRA + PX)}개', flush=True)
res = {k: pd.Series(np.nan, index=TA.index) for k in ['ALL', 'ALL−px']}
for dd in DATES:
    fm = (TA.date != dd).values; f, v = TA[fm], TA[~fm]
    f, v = G['add_te'](f, v)                                   # 구간 중앙값: 학습 날짜로, 학습 행은 자기 날짜 빼고
    mf = np.full(len(f), np.nan)
    for d2 in f.date.unique():
        m = (f.date == d2).values; mf[m] = seg_med(f[~m], f[m])
    mv = seg_med(f, v)
    f, v = add_est(f, mf), add_est(v, mv)
    f, v = add_minute(f, mf), add_minute(v, mv)
    for name, feats in [('ALL', BASEF + TE + EXTRA + PX), ('ALL−px', BASEF + TE + EXTRA)]:
        res[name][v.index] = lgb.LGBMRegressor(**LGB).fit(to_X(f, feats), f.tt_sec).predict(to_X(v, feats))
    print(f'fold {dd} 끝 ({time.time() - t0:.0f}초)', flush=True)

old = pd.read_parquet(f'{ROOT}/experiments/oof_final_E41.parquet').set_index('row_id').loc[TA.row_id.values]
rows = []
for n, p in [('E41 LightGBM', old.lgb.values), ('ALL', res['ALL'].values), ('ALL−px', res['ALL−px'].values)]:
    r = cv_of(pd.Series(p, index=TA.index))
    rows.append({'실험': n, '월': r['월'], '화': r['화'], '수': r['수'], '토': r['토'], '평일': (r['월'] + r['화'] + r['수']) / 3, '예상': r['예상']})
T = pd.DataFrame(rows).set_index('실험').round(3)
print(T.to_string(), flush=True)
T.to_csv(f'{ROOT}/experiments/E72_cv_table.csv')
pd.DataFrame({'row_id': TA.row_id.values, 'all': res['ALL'].values, 'all_nopx': res['ALL−px'].values}).to_parquet(f'{ROOT}/experiments/E72_oof.parquet')
