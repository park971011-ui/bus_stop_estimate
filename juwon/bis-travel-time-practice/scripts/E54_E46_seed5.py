"""E54 = E46과 같은 방법인데 모든 모델을 seed 0~4로 5번 학습해서 평균 (그 밖엔 전부 E46과 같음)
  - E41 부분: LightGBM(E30) seed 5개 평균 × 0.8 + CatBoost(E31) seed 5개 평균 × 0.2
  - 목·금: E41 부분 × 0.5 + 평일 전용 LightGBM seed 5개 평균 × 0.5
  - 일요일: E45 규칙 (배차 같은 노선 토처럼·평일처럼 반반, 다른 노선 토처럼·평일처럼·그대로 1/3씩)
  seed 평균은 예측의 흔들림만 줄임 (검증: LightGBM seed 5개 −0.015, E36 연습 −0.016)
실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E54_E46_seed5.py practice   (또는 final)
결과: submissions/{practice|final}_E54_E46_seed5.csv (각 약 10분). practice면 E46과 연습 test 비교
주의: 노트북 cv_feature_experiments.ipynb 의 셀(2,4,6,8,10,27,29,41번)을 실행해서 피처·모델 함수를 가져옴. E46 파일은 안 건드림.
"""
import json, os, re, sys, time
import numpy as np, pandas as pd, lightgbm as lgb
os.chdir('/Users/kimjuwon/Desktop/에모팀플/bis-travel-time-practice/notebooks')
PH = sys.argv[1]
nb = json.load(open('cv_feature_experiments.ipynb')); G = {}
for i in [2, 4, 6, 8, 10, 27, 29, 41]:
    src = ''.join(nb['cells'][i]['source'])
    if i == 2:
        src = re.sub(r"PHASE = '\w+'", f"PHASE = '{PH}'", src, count=1); assert f"PHASE = '{PH}'" in src
    exec(src, G)
TRAIN, TEST, test, to_X, DAY = G['TRAIN'], G['TEST'], G['test'], G['to_X'], G['DAY']
assert G['PHASE'] == PH
SEGK = ['route_id', 'direction', 'from_stop', 'to_stop']


def turn_of(fit):
    """학습 날짜(fit)만으로 회차 구간: 노선마다 상행 구간 중 중앙값 최대"""
    med = fit[fit.direction == 0].groupby(SEGK).tt_sec.median().reset_index()
    return med.loc[med.groupby('route_id').tt_sec.idxmax(), SEGK].assign(is_turn2=1)


TRIP = ['date', 'vehicle_id', 'route_id', 'direction', 'trip_no']
VD = ['date', 'vehicle_id']          # 차량 하루 (확인: 차량 하루에 노선은 항상 1개)
H = ['t_prev_gap', 't_blk_n', 't_blk_dist', 't_blk_before', 't_blk_after',
     'v_blk_n', 'v_prev_n', 'v_next_n', 'v_next_trip_gap', 'v_next_trip_n0']


def add_turn_hint(df, TURN):
    """회차 구간은 상행의 마지막(또는 끝에서 둘째) 구간이라, 바로 뒤 행은 반대 방향 다음 운행에 있음 → 차량 하루 단위로 봄."""
    df = df.merge(TURN, on=SEGK, how='left').set_index(df.index)
    df['is_turn2'] = df.is_turn2.fillna(0).astype(int)
    s = df.sort_values(TRIP + ['stop_seq']).copy()
    s['t_prev_gap'] = s.arrive_hour - s.groupby(TRIP).arrive_hour.shift(1)     # 같은 운행 바로 앞 구간과 시각 차이
    b = s.groupby(TRIP + ['arrive_hour'])
    s['t_blk_n'] = b.stop_seq.transform('size')                                # 이 운행에서 같은 시에 든 구간 수
    s['t_blk_dist'] = b.dist_m.transform('sum')
    s['t_blk_before'] = s.stop_seq - b.stop_seq.transform('min')
    s['t_blk_after'] = b.stop_seq.transform('max') - s.stop_seq
    # 차량 하루 전체에서 시간별 행 수 (방향 상관없이)
    vc = s.groupby(VD + ['arrive_hour']).size()
    def vcount(shift):
        k = pd.MultiIndex.from_arrays([s.date, s.vehicle_id, s.arrive_hour + shift])
        return vc.reindex(k).fillna(0).values
    s['v_blk_n'], s['v_prev_n'], s['v_next_n'] = vcount(0), vcount(-1), vcount(1)
    # 다음 운행: 이 운행이 끝난 시각 이후에 시작하는 같은 차량의 운행 중 가장 이른 것
    tr = s.groupby(TRIP).agg(start=('arrive_hour', 'min'), end=('arrive_hour', 'max'), first_seq=('stop_seq', 'min')).reset_index()
    first_n = s.merge(tr[TRIP + ['start']], on=TRIP).query('arrive_hour == start').groupby(TRIP).size().rename('n0').reset_index()
    tr = tr.merge(first_n, on=TRIP, how='left')
    nxt = []
    for _, g in tr.groupby(VD):
        g = g.sort_values(['start', 'end'])
        for r in g.itertuples():
            c = g[(g.start >= r.end) & ~((g.direction == r.direction) & (g.trip_no == r.trip_no) & (g.route_id == r.route_id))]
            c = c[c.direction != r.direction] if (c.direction != r.direction).any() else c
            nxt.append((r.date, r.vehicle_id, r.route_id, r.direction, r.trip_no,
                        c.start.min() if len(c) else np.nan, c.sort_values('start').n0.iloc[0] if len(c) else np.nan))
    nxt = pd.DataFrame(nxt, columns=TRIP + ['next_start', 'v_next_trip_n0'])
    s = s.merge(nxt, on=TRIP, how='left').set_index(s.index)
    s['v_next_trip_gap'] = s.next_start - s.arrive_hour                         # 다음 운행 시작까지 몇 시간
    s.loc[s.is_turn2 == 0, H] = np.nan                                           # 회차 구간 행에만 값
    return s.loc[df.index]

from catboost import CatBoostRegressor
PH = sys.argv[1]
F = G['feats_of'](['base', 'seg', 'time', 'trip']) + H
LGB = {**G['PARAMS'], 'min_child_samples': 10, 'learning_rate': 0.02, 'n_estimators': 1500}
CB = dict(loss_function='MAE', iterations=2000, learning_rate=0.08, depth=10, l2_leaf_reg=3, random_seed=0, verbose=0, thread_count=-1)
N = 5
dow = lambda df: pd.to_datetime(df.date).dt.day_of_week.values
t0 = time.time()
TURN_ALL = turn_of(TRAIN)
TR, TE = add_turn_hint(TRAIN, TURN_ALL), add_turn_hint(TEST, TURN_ALL)
sun = dow(TE) == 6
S = TE.copy(); S['is_weekend'] = 1; S['interval'] = S.interval_sat
W = TE.copy(); W['is_weekend'] = 0; W['interval'] = W.interval_wd
acc = {k: [] for k in ['base', 'sat', 'wd']}
for sd in range(N):
    ml = lgb.LGBMRegressor(**{**LGB, 'random_state': sd}).fit(to_X(TR, F), TR.tt_sec)
    mc = CatBoostRegressor(**{**CB, 'random_seed': sd}, cat_features=['route_id', 'seg_id']).fit(TR[F], TR.tt_sec)
    for k, X in [('base', TE), ('sat', S), ('wd', W)]:
        acc[k].append(0.8 * ml.predict(to_X(X, F)) + 0.2 * mc.predict(X[F]))
    print(f'[{PH}] E41 부분 seed {sd} ({time.time() - t0:.0f}초)', flush=True)
pB, pS, pW = (np.mean(acc[k], axis=0) for k in ['base', 'sat', 'wd'])
diff = (TE.interval_sun != TE.interval_sat).values
p_sun = np.where(diff, (pS + pW + pB) / 3, 0.5 * pS + 0.5 * pW)
WD = TRAIN[dow(TRAIN) < 5]; TW = turn_of(WD)
FW, TW_ = add_turn_hint(WD, TW), add_turn_hint(TEST, TW)
pwd = np.mean([lgb.LGBMRegressor(**{**LGB, 'random_state': sd}).fit(to_X(FW, F), FW.tt_sec).predict(to_X(TW_, F)) for sd in range(N)], axis=0)
print(f'[{PH}] 평일 전용 seed {N}개 끝 ({time.time() - t0:.0f}초)', flush=True)
p = np.round(np.where(sun, p_sun, 0.5 * pB + 0.5 * pwd), 1)
sub = pd.DataFrame({'row_id': TE.row_id.values, 'tt_sec': p}).set_index('row_id').loc[test.row_id].reset_index()
assert len(sub) == len(test) and sub.tt_sec.notna().all() and not sub.row_id.duplicated().any()
if os.path.exists(f'../data/{PH}/sample_submission.csv'):
    ss = pd.read_csv(f'../data/{PH}/sample_submission.csv'); assert (sub.row_id.values == ss.row_id.values).all()
print(f'[{PH}] 음수 예측 {(sub.tt_sec < 0).sum()}개', flush=True)
assert PH != 'final' or (sub.tt_sec >= 0).all()
path = f'../submissions/{PH}_E54_E46_seed5.csv'; sub.to_csv(path, index=False); print(f'[{PH}] 저장 {path} {sub.shape}', flush=True)
e46 = pd.read_csv(f'../submissions/{PH}_E46_wd_route_sunmix.csv').set_index('row_id').tt_sec.loc[test.row_id].values
dw = pd.to_datetime(test.date).dt.day_of_week.values
print(f'[{PH}] E46 대비 평균 |차이| {np.abs(sub.tt_sec.values - e46).mean():.2f}초', flush=True)
if 'tt_sec' in test.columns:
    y = test.tt_sec.values
    for n, q in [('E46', e46), ('E54', sub.tt_sec.values)]:
        a = np.abs(y - q); print(f'[{PH}] {n} 연습 test {a.mean():.3f} | 목·금 {a[dw < 5].mean():.3f} | 일 {a[dw == 6].mean():.3f}', flush=True)
