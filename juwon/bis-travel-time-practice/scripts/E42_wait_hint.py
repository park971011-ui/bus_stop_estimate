"""E42 = E41의 회차 힌트를 "대기 구간 전체"로 넓힘 (LightGBM 0.8 + CatBoost 0.2, 일요일 반반은 E41과 같음)

대기 구간: train에서 그 구간의 "평소(중앙값)와의 평균 차이"가 TH초 이상인 구간 + 회차 구간 (train만 보고 고름)
  - 회차 말고도 노선 중간 기점(예: 2번 18번째 구간, 중앙값 440초)이나 출발 직후 구간에서 오래 기다리는 곳이 있음
  - TH = 60 이면 대회 구간 435개 중 17개 (행 3.5%), 80 이면 9개
힌트 (입력 arrive_hour만 사용, 정답 안 씀, 대기 구간 행에만 값): E41의 10개 + 같은 운행 바로 다음 구간과의 시각 차이, 다음 시간 칸의 구간 수
검증(cv)에서는 대기 구간을 fold마다 학습 날짜만으로 다시 고름

실행: cd bis-travel-time-practice
  /opt/miniconda3/envs/bis/bin/python scripts/E42_wait_hint.py final cv          → 대회 train 4-fold 검증 (E41과 비교, 약 25분), experiments/oof_final_E42.parquet
  /opt/miniconda3/envs/bis/bin/python scripts/E42_wait_hint.py practice make 60  → submissions/practice_E42_waithint_sunmix.csv + 연습 test 점수
  /opt/miniconda3/envs/bis/bin/python scripts/E42_wait_hint.py final make 60     → submissions/final_E42_waithint_sunmix.csv
주의: 노트북 cv_feature_experiments.ipynb 의 셀(2,4,6,8,10,27,29,41번)을 실행해서 피처·모델 함수를 가져옴.
"""
import json, os, re, sys, time
import numpy as np, pandas as pd, lightgbm as lgb
from catboost import CatBoostRegressor
os.chdir('/Users/kimjuwon/Desktop/에모팀플/bis-travel-time-practice/notebooks')
PH, MODE = sys.argv[1], sys.argv[2]
nb = json.load(open('cv_feature_experiments.ipynb')); G = {}
for i in [2, 4, 6, 8, 10, 27, 29, 41]:
    src = ''.join(nb['cells'][i]['source'])
    if i == 2:
        src = re.sub(r"PHASE = '\w+'", f"PHASE = '{PH}'", src, count=1); assert f"PHASE = '{PH}'" in src
    exec(src, G)
TRAIN, TEST, test, to_X = G['TRAIN'], G['TEST'], G['test'], G['to_X']
assert G['PHASE'] == PH

SEGK = ['route_id', 'direction', 'from_stop', 'to_stop']
TRIP = ['date', 'vehicle_id', 'route_id', 'direction', 'trip_no']
VD = ['date', 'vehicle_id']
H = ['t_prev_gap', 't_next_gap', 't_blk_n', 't_blk_dist', 't_blk_before', 't_blk_after', 't_next_blk_n',
     'v_blk_n', 'v_prev_n', 'v_next_n', 'v_next_trip_gap', 'v_next_trip_n0']


def hint_raw(df):
    """모든 행에 힌트 계산 (나중에 대기 구간 행만 남김). E41 add_turn_hint와 같은 계산 + 다음 구간 2개."""
    s = df.sort_values(TRIP + ['stop_seq']).copy()
    g = s.groupby(TRIP).arrive_hour
    s['t_prev_gap'] = s.arrive_hour - g.shift(1)
    s['t_next_gap'] = g.shift(-1) - s.arrive_hour                               # 같은 운행 바로 다음 구간과 시각 차이
    b = s.groupby(TRIP + ['arrive_hour'])
    s['t_blk_n'] = b.stop_seq.transform('size')
    s['t_blk_dist'] = b.dist_m.transform('sum')
    s['t_blk_before'] = s.stop_seq - b.stop_seq.transform('min')
    s['t_blk_after'] = b.stop_seq.transform('max') - s.stop_seq
    tc = s.groupby(TRIP + ['arrive_hour']).size()
    k = pd.MultiIndex.from_arrays([s[c] for c in TRIP] + [s.arrive_hour + 1])
    s['t_next_blk_n'] = tc.reindex(k).fillna(0).values                           # 같은 운행 다음 시간 칸의 구간 수
    vc = s.groupby(VD + ['arrive_hour']).size()
    def vcount(shift):
        k = pd.MultiIndex.from_arrays([s.date, s.vehicle_id, s.arrive_hour + shift])
        return vc.reindex(k).fillna(0).values
    s['v_blk_n'], s['v_prev_n'], s['v_next_n'] = vcount(0), vcount(-1), vcount(1)
    tr = s.groupby(TRIP).agg(start=('arrive_hour', 'min'), end=('arrive_hour', 'max')).reset_index()
    first_n = s.merge(tr[TRIP + ['start']], on=TRIP).query('arrive_hour == start').groupby(TRIP).size().rename('n0').reset_index()
    tr = tr.merge(first_n, on=TRIP, how='left')
    nxt = []
    for _, gg in tr.groupby(VD):
        gg = gg.sort_values(['start', 'end'])
        for r in gg.itertuples():
            c = gg[(gg.start >= r.end) & ~((gg.direction == r.direction) & (gg.trip_no == r.trip_no) & (gg.route_id == r.route_id))]
            c = c[c.direction != r.direction] if (c.direction != r.direction).any() else c
            nxt.append((r.date, r.vehicle_id, r.route_id, r.direction, r.trip_no,
                        c.start.min() if len(c) else np.nan, c.sort_values('start').n0.iloc[0] if len(c) else np.nan))
    nxt = pd.DataFrame(nxt, columns=TRIP + ['next_start', 'v_next_trip_n0'])
    s = s.merge(nxt, on=TRIP, how='left').set_index(s.index)
    s['v_next_trip_gap'] = s.next_start - s.arrive_hour
    return s.loc[df.index]


def wait_segs(fit, th):
    """fit(학습 행)만 보고 대기 구간 고르기: 평소와의 평균 차이 >= th초 + 회차 구간(상행 중앙값 최대)"""
    f = fit[SEGK + ['tt_sec']].copy()
    f['dev'] = (f.tt_sec - f.groupby(SEGK).tt_sec.transform('median')).abs()
    sp = f.groupby(SEGK).dev.mean()
    w = set(sp[sp >= th].index)
    m0 = f[f.direction == 0].groupby(SEGK).tt_sec.median().reset_index()
    w |= set(map(tuple, m0.loc[m0.groupby('route_id').tt_sec.idxmax(), SEGK].values))
    return w


def masked(df, segs):
    df = df.copy()
    on = pd.Series(list(zip(*[df[c] for c in SEGK])), index=df.index).isin(segs).values
    df['is_wait'] = on.astype(int)
    df.loc[~on, H] = np.nan
    return df


RAW_TR, RAW_TE = hint_raw(TRAIN), hint_raw(TEST)
F0 = G['feats_of'](['base', 'seg', 'time', 'trip'])
F = F0 + H
LGB = {**G['PARAMS'], 'min_child_samples': 10, 'learning_rate': 0.02, 'n_estimators': 1500}
CB = dict(loss_function='MAE', iterations=2000, learning_rate=0.08, depth=10, l2_leaf_reg=3, random_seed=0, verbose=0, thread_count=-1)
fit_lgb = lambda f: lgb.LGBMRegressor(**LGB).fit(to_X(f, F), f.tt_sec)
fit_cb = lambda f: CatBoostRegressor(**CB, cat_features=['route_id', 'seg_id']).fit(f[F], f.tt_sec)

if MODE == 'cv':
    DATES, DAY, cv_of = G['DATES'], G['DAY'], G['cv_of']
    old = pd.read_parquet('../experiments/oof_final_E41.parquet').set_index('row_id').loc[TRAIN.row_id.values]
    y = TRAIN.tt_sec.values
    allw = masked(RAW_TR, wait_segs(TRAIN, 60)).is_wait.values == 1          # 보고용 (train 전체 기준 대기 구간 60)
    fmt = lambda r: ' | '.join(f'{k} {r[k]:.3f}' for k in ['월', '화', '수', '토', '평일', '예상'])
    def rep(name, p):
        r = cv_of(pd.Series(p, index=TRAIN.index))
        print(f'[{PH}] {name:18s} {fmt(r)} | 대기 구간 행 {np.abs(y - p)[allw].mean():.1f} | 나머지 {np.abs(y - p)[~allw].mean():.3f}', flush=True)
        return r['예상']
    print(f'[{PH}] 대기 구간(train 전체, 60초): {len(wait_segs(TRAIN, 60))}개, 행 {allw.mean():.1%} | 80초: {len(wait_segs(TRAIN, 80))}개', flush=True)
    rep('E41 LightGBM', old.lgb.values)
    t0 = time.time(); best = None
    for th in [60, 80]:
        o = pd.Series(np.nan, index=TRAIN.index)
        for d in DATES:
            fm = TRAIN.date != d; segs = wait_segs(TRAIN[fm], th)
            f, v = masked(RAW_TR[fm], segs), masked(RAW_TR[~fm], segs)
            o[v.index] = fit_lgb(f).predict(to_X(v, F))
        s = rep(f'E42 LightGBM {th}초', o.values); print(f'    ({time.time() - t0:.0f}초)', flush=True)
        if best is None or s < best[0]: best = (s, th, o)
    _, th, ol = best
    print(f'[{PH}] LightGBM 기준 더 좋은 쪽: {th}초 → CatBoost도 학습', flush=True)
    oc = pd.Series(np.nan, index=TRAIN.index)
    for d in DATES:
        fm = TRAIN.date != d; segs = wait_segs(TRAIN[fm], th)
        oc[TRAIN.index[~fm]] = fit_cb(masked(RAW_TR[fm], segs)).predict(masked(RAW_TR[~fm], segs)[F])
    rep('E41 CatBoost', old.cb.values); rep(f'E42 CatBoost {th}초', oc.values)
    rep('E41 섞기 0.8', old.e41.values); rep(f'E42 섞기 0.8 {th}초', 0.8 * ol.values + 0.2 * oc.values)
    pd.DataFrame({'row_id': TRAIN.row_id.values, 'lgb': ol.values, 'cb': oc.values, 'e42': 0.8 * ol.values + 0.2 * oc.values}).to_parquet('../experiments/oof_final_E42.parquet')
    print(f'[{PH}] 저장 ../experiments/oof_final_E42.parquet | 대기 기준 {th}초 ({time.time() - t0:.0f}초)')
else:
    th = int(sys.argv[3])
    segs = wait_segs(TRAIN, th)
    TR, TE = masked(RAW_TR, segs), masked(RAW_TE, segs)
    print(f'[{PH}] 대기 구간 {len(segs)}개 ({th}초) | 행: train {TR.is_wait.mean():.1%} test {TE.is_wait.mean():.1%}', flush=True)
    t0 = time.time(); ml, mc = fit_lgb(TR), fit_cb(TR)
    pred = lambda X: 0.8 * ml.predict(to_X(X, F)) + 0.2 * mc.predict(X[F])
    sun = pd.to_datetime(TE.date).dt.day_of_week.values == 6
    S = TE.copy(); S['is_weekend'] = 1; S['interval'] = S.interval_sat
    W = TE.copy(); W['is_weekend'] = 0; W['interval'] = W.interval_wd
    p = np.where(sun, 0.5 * pred(S) + 0.5 * pred(W), pred(TE))
    print(f'[{PH}] 학습·예측 {time.time() - t0:.0f}초', flush=True)
    sub = pd.DataFrame({'row_id': TE.row_id.values, 'tt_sec': np.round(p, 1)}).set_index('row_id').loc[test.row_id].reset_index()
    assert len(sub) == len(test) and sub.tt_sec.notna().all() and not sub.row_id.duplicated().any()
    path = f'../submissions/{PH}_E42_waithint_sunmix.csv'; sub.to_csv(path, index=False)
    print(f'[{PH}] 저장 {path} {sub.shape}', flush=True)
    old = pd.read_csv(f'../submissions/{PH}_E41_turnhint_sunmix.csv').set_index('row_id').tt_sec.loc[test.row_id].values
    t = TE.set_index('row_id').loc[test.row_id]; w = t.is_wait.values == 1
    sn = pd.to_datetime(t.date).dt.day_of_week.values == 6; new = sub.tt_sec.values
    print(f'[{PH}] E41 대비 예측 바뀐 정도: 대기 구간 행 {np.abs(new - old)[w].mean():.1f}초 | 나머지 {np.abs(new - old)[~w].mean():.2f}초', flush=True)
    if 'tt_sec' in test.columns:
        y = test.tt_sec.values
        for name, q in [('E41', old), ('E42', new)]:
            a = np.abs(y - q)
            print(f'[{PH}] {name} 연습 test {a.mean():.3f} | 목금 {a[~sn].mean():.3f} | 일 {a[sn].mean():.3f} | 대기 구간 행 {a[w].mean():.1f} | 나머지 {a[~w].mean():.3f}', flush=True)
