"""E56 = E46 방법 그대로 + 세 모델(E41 LightGBM·CatBoost, 평일 전용 LightGBM) 모두에 E55 "추정 회차 시간" 피처 추가
  추정 회차 시간 = 3574 − 같은 날·차량·시간대 다른 구간들의 평소 시간(구간 중앙값) 합, 회차 행에만 (자세히 E55_turn_time_est.py)
    - 예측할 행: 학습 행 전체로 중앙값 / 학습 행: 자기 날짜를 뺀 다른 학습 날짜로
  나머지(섞는 비율, seed 0, 일요일 E45 규칙, 목·금 E41 × 0.5 + 평일 전용 × 0.5)는 E46과 같음. 단 목·금의 E41 값은 반올림 전 값 사용.
실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E56_E46_turnest.py practice   (또는 final)
결과: submissions/{practice|final}_E56_E46_turnest.csv. practice면 E46과 연습 test 비교
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

def seg_med(src, apply):
    md = src.groupby(SEGK).tt_sec.median()
    return apply[SEGK].join(md.rename('x'), on=SEGK).x.values


def add_est(df, med):
    df = df.copy(); df['_m'] = med
    hs = df.groupby(V + ['arrive_hour'])._m.sum()
    k = lambda s: pd.MultiIndex.from_arrays([df.date, df.vehicle_id, df.arrive_hour + s])
    inner = k(-1).isin(hs.index) & k(1).isin(hs.index)
    others = hs.reindex(k(0)).values - np.nan_to_num(df._m.values)
    df['turn_est'] = np.where((df.is_turn2 == 1) & inner, C - others, np.nan)
    return df.drop(columns='_m')


from catboost import CatBoostRegressor
F = G['feats_of'](['base', 'seg', 'time', 'trip']) + H
FE = F + ['turn_est']
LGB = {**G['PARAMS'], 'min_child_samples': 10, 'learning_rate': 0.02, 'n_estimators': 1500}
CB = dict(loss_function='MAE', iterations=2000, learning_rate=0.08, depth=10, l2_leaf_reg=3, random_seed=0, verbose=0, thread_count=-1)
V = ['date', 'vehicle_id']; C = 3574.0
dow = lambda df: pd.to_datetime(df.date).dt.day_of_week.values


def prep(fit_raw, apply_raw):
    turn = turn_of(fit_raw)
    f, a = add_turn_hint(fit_raw, turn), add_turn_hint(apply_raw, turn)
    mf = np.full(len(f), np.nan)
    for dd in f.date.unique():
        m = (f.date == dd).values; mf[m] = seg_med(f[~m], f[m])
    return add_est(f, mf), add_est(a, seg_med(f, a))


t0 = time.time()
f, a = prep(TRAIN, TEST)
sun = dow(a) == 6
S = a.copy(); S['is_weekend'] = 1; S['interval'] = S.interval_sat
W = a.copy(); W['is_weekend'] = 0; W['interval'] = W.interval_wd
ml = lgb.LGBMRegressor(**LGB).fit(to_X(f, FE), f.tt_sec)
mc = CatBoostRegressor(**CB, cat_features=['route_id', 'seg_id']).fit(f[FE], f.tt_sec)
pr = lambda X: 0.8 * ml.predict(to_X(X, FE)) + 0.2 * mc.predict(X[FE])
pB, pS, pW = pr(a), pr(S), pr(W)
diff = (a.interval_sun != a.interval_sat).values
p_sun = np.where(diff, (pS + pW + pB) / 3, 0.5 * pS + 0.5 * pW)
print(f'[{PH}] E41 부분 ({time.time() - t0:.0f}초) | test 회차 행 중 추정값 있음 {a.turn_est.notna().sum()}/{(a.is_turn2 == 1).sum()}', flush=True)
fw, aw = prep(TRAIN[dow(TRAIN) < 5], TEST)
pwd = lgb.LGBMRegressor(**LGB).fit(to_X(fw, FE), fw.tt_sec).predict(to_X(aw, FE))
print(f'[{PH}] 평일 전용 ({time.time() - t0:.0f}초)', flush=True)
p = np.round(np.where(sun, p_sun, 0.5 * pB + 0.5 * pwd), 1)
sub = pd.DataFrame({'row_id': a.row_id.values, 'tt_sec': p}).set_index('row_id').loc[test.row_id].reset_index()
assert len(sub) == len(test) and sub.tt_sec.notna().all() and not sub.row_id.duplicated().any()
if os.path.exists(f'../data/{PH}/sample_submission.csv'):
    ss = pd.read_csv(f'../data/{PH}/sample_submission.csv'); assert len(sub) == len(ss) and (sub.row_id.values == ss.row_id.values).all()
print(f'[{PH}] 음수 예측 {(sub.tt_sec < 0).sum()}개', flush=True)
assert PH != 'final' or (sub.tt_sec >= 0).all()
path = f'../submissions/{PH}_E56_E46_turnest.csv'; assert not os.path.exists(path); sub.to_csv(path, index=False)
print(f'[{PH}] 저장 {path} {sub.shape}', flush=True)
e46 = pd.read_csv(f'../submissions/{PH}_E46_wd_route_sunmix.csv').set_index('row_id').tt_sec.loc[test.row_id].values
t = a.set_index('row_id').loc[test.row_id]; tr = t.is_turn2.values == 1; dw = dow(t)
print(f'[{PH}] E46 대비 평균 |차이|: 회차 행 {np.abs(sub.tt_sec.values - e46)[tr].mean():.1f}초 | 나머지 {np.abs(sub.tt_sec.values - e46)[~tr].mean():.2f}초', flush=True)
if 'tt_sec' in test.columns:
    y = test.tt_sec.values
    for n, q in [('E46', e46), ('E56', sub.tt_sec.values)]:
        e = np.abs(y - q)
        print(f'[{PH}] {n} 연습 test {e.mean():.3f} | 목·금 {e[dw < 5].mean():.3f} | 일 {e[dw == 6].mean():.3f} | 회차 행 {e[tr].mean():.1f} | 나머지 {e[~tr].mean():.3f}', flush=True)
