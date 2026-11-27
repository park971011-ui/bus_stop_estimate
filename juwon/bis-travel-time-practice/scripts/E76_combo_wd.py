"""E76 목·금 = E56 방식(기본 LGB 0.8 + CB 0.2 를 0.5, 평일 전용 LGB 0.5) + E74 피처 조합 (peak2, wkhour, turn 추가, turn_est는 원래 있음)
  일요일 = E73 파일 그대로 (일요일 모델 + 같은 날 연습 기록)
  근거: E74 대회 train 4-fold 25.227 → 25.113 (4일 모두 개선). 단 E75 연습 목·금 확인에서는 26.102 → 26.142 (+0.040, 나빠짐)
        → 엇갈려서 public으로 확인 (E73 단독과 비교하면 목·금 조합 효과만 보임)
실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E76_combo_wd.py final
결과: submissions/final_E76_combo_wd_E73sun.csv
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
FE = F + ['turn_est'] + G['FEATS']['peak2'] + G['FEATS']['wkhour'] + G['FEATS']['turn']   # E74 조합 43개
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
p = np.round(0.5 * pB + 0.5 * pwd, 1)
assert PH == 'final'
new = pd.read_csv('../submissions/final_E73_sunday_px.csv').set_index('row_id').tt_sec
old = new.copy()
ids = a.row_id.values[~sun]; new.loc[ids] = p[~sun]                          # 목·금만 교체, 일요일은 E73 그대로
ss = pd.read_csv('../data/final/sample_submission.csv')
sub = new.loc[ss.row_id].reset_index(); sub.columns = ['row_id', 'tt_sec']
assert len(sub) == 111563 and sub.tt_sec.notna().all() and not sub.row_id.duplicated().any() and (sub.tt_sec >= 0).all()
path = '../submissions/final_E76_combo_wd_E73sun.csv'; assert not os.path.exists(path); sub.to_csv(path, index=False)
d = new.loc[ids].values - old.loc[ids].values
print(f'[{PH}] 저장 {path} | 일요일 = E73 | 목·금 E73(=E56) 대비 평균 {d.mean():+.2f}초, |차이| {np.abs(d).mean():.2f}초 ({time.time() - t0:.0f}초)', flush=True)
