"""E101 = E67 + 목·금에 평일 전용 ExtraTrees(criterion='absolute_error') 20% 섞기 (다윤 extratrees_mae_experiment 재현)
  ET: 월·화·수만 학습, 피처 E56의 35개(turn_est 제외, 회차 힌트 H 포함), 결측 0, 범주형은 숫자 코드
      n_estimators=100, min_samples_leaf=10, max_features=0.6, random_state=0
  목·금 = 0.8 × E67(=E56 목·금) + 0.2 × ET, 일요일은 E67 그대로
  cv:    평일 3-fold(월·화·수 하루씩 빼기)로 다윤 결과(기준 23.235 → 20% 섞기 23.222)가 나오는지 확인
  final: submissions/final_E101_E67_et_mae20.csv
실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E101_et_mae_wd.py cv   (또는 final)
"""
import json, os, re, sys, time
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

MODE = sys.argv[1]; assert MODE in ('cv', 'final')
NT = int(sys.argv[2]) if len(sys.argv) > 2 else 100
ROOT = '/Users/kimjuwon/Desktop/에모팀플/bis-travel-time-practice'
# E56 스크립트의 준비 부분(prep, F 등)만 가져옴: 모델 학습 전까지
src = open(f'{ROOT}/scripts/E56_E46_turnest.py').read().split('\nt0 = time.time()')[0]
sys.argv = [sys.argv[0], 'final']
exec(src)
assert len(F) == 35

ET = dict(n_estimators=NT, criterion='absolute_error', min_samples_leaf=10, max_features=0.6, random_state=0, n_jobs=-1)


def to_num(df):
    x = df[F].copy()
    for c in ['route_id', 'seg_id']:
        x[c] = x[c].astype(G['CAT_DTYPE'][c]).cat.codes
    return x.astype(float).fillna(0).values


def et_fit_predict(fit_raw, apply_raw):
    f, a = prep(fit_raw, apply_raw)
    m = ExtraTreesRegressor(**ET).fit(to_num(f), f.tt_sec.values)
    return pd.Series(m.predict(to_num(a)), index=a.index)


t0 = time.time()
WD = TRAIN[dow(TRAIN) < 5]
if MODE == 'cv':
    o = pd.read_parquet(f'{ROOT}/experiments/oof_final_E56.parquet').set_index('row_id').loc[WD.row_id.values]
    base = pd.Series(0.5 * o.e56_41part.values + 0.5 * o.e56_wd_lgb.values, index=WD.index)
    et = pd.Series(np.nan, index=WD.index)
    for dd in sorted(WD.date.unique()):
        m = (WD.date == dd).values
        et[WD.index[m]] = et_fit_predict(WD[~m], WD[m]).loc[WD.index[m]].values
        print(f'[cv] {dd} 끝 ({time.time() - t0:.0f}초)', flush=True)
    y = WD.tt_sec
    rows = []
    for name, p in [('기준 E56 평일', base), ('ET 단독', et), ('기준 90% + ET 10%', 0.9 * base + 0.1 * et), ('기준 80% + ET 20%', 0.8 * base + 0.2 * et)]:
        e = (y - p).abs()
        rows.append({'예측': name, **{DAY[pd.Timestamp(dd).day_of_week]: e[WD.date == dd].mean() for dd in sorted(WD.date.unique())},
                     'MAE': e.mean(), 'RMSE': np.sqrt(((y - p) ** 2).mean())})
    r = pd.DataFrame(rows).set_index('예측'); r['ΔMAE'] = r.MAE - r.MAE.iloc[0]
    print(r.round(4).to_string(), flush=True)
    pd.DataFrame({'row_id': WD.row_id.values, 'et_mae': et.values}).to_parquet(f'{ROOT}/experiments/E101_et_mae_oof.parquet')
else:
    p_et = et_fit_predict(WD, TEST)
    a = TEST.loc[p_et.index]
    e67 = pd.read_csv(f'{ROOT}/submissions/final_E67_sunday_weight_route.csv').set_index('row_id').tt_sec
    et = pd.Series(p_et.values, index=a.row_id.values)
    sub = e67.loc[test.row_id].copy()
    thufri = dow(test) < 5
    ids = test.row_id.values[thufri]
    sub.loc[ids] = np.round(0.8 * e67.loc[ids].values + 0.2 * et.loc[ids].values, 1)
    sub = sub.reset_index()
    ss = pd.read_csv(f'{ROOT}/data/final/sample_submission.csv')
    assert len(sub) == len(ss) and (sub.row_id.values == ss.row_id.values).all() and sub.tt_sec.notna().all() and (sub.tt_sec >= 0).all()
    d = (sub.tt_sec.values - e67.loc[test.row_id].values)
    print(f'[final] 목·금 {thufri.sum()}행 바뀜 | 평균 차이 {d[thufri].mean():+.2f}초 | 평균 |차이| {np.abs(d[thufri]).mean():.2f}초 | 일요일 바뀐 행 {(d[~thufri] != 0).sum()}', flush=True)
    path = f'{ROOT}/submissions/final_E101_E67_et_mae20.csv'; assert not os.path.exists(path)
    sub.to_csv(path, index=False)
    print(f'[final] 저장 {path} {sub.shape} ({time.time() - t0:.0f}초)', flush=True)
