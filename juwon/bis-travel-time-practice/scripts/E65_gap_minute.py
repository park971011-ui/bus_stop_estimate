"""E65 새 단서 2개를 E41 LightGBM에 각각 넣어 대회 train 날짜별 4-fold 검증 (E41 LightGBM 25.227과 비교)
  A) 기록 빠짐: 같은 운행 안에서 바로 앞 행과 순번이 몇 개 건너뛰었나(gap_before), 바로 뒤 행과(gap_after)
     - 행 순서(row_id) = 차량 → 정류장 순서. 운행 1%에서 기록이 빠져 있고, 빠진 직후 행은 MAE 43.8 (보통 24.6)
  B) 분 단위 위치 추정(est_min): 차량 하루 기록에서 시(arrive_hour)가 바뀌는 지점을 정각으로 보고,
     구간 평소 시간(자기 날짜 뺀 중앙값)을 더하거나 빼서 몇 분쯤인지 추정 + 연속 시각 est_time = 시 + 분/60
  입력만 씀 (정답은 평소 시간 계산에서 그 행 날짜를 빼고만 씀)
실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E65_gap_minute.py final
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

F = G['feats_of'](['base', 'seg', 'time', 'trip']) + H
LGB = {**G['PARAMS'], 'min_child_samples': 10, 'learning_rate': 0.02, 'n_estimators': 1500}
DATES, cv_of = G['DATES'], G['cv_of']
V = ['date', 'vehicle_id']; C = 3574.0
TURN_ALL = turn_of(TRAIN)                       # E41과 같음 (train 전체로 고른 회차 구간)
TR = add_turn_hint(TRAIN, TURN_ALL)


F = G['feats_of'](['base', 'seg', 'time', 'trip']) + H
LGB = {**G['PARAMS'], 'min_child_samples': 10, 'learning_rate': 0.02, 'n_estimators': 1500}
DATES, cv_of = G['DATES'], G['cv_of']
TR = add_turn_hint(TRAIN, turn_of(TRAIN))
V = ['date', 'vehicle_id']


def add_gap(df):
    d = df.copy(); d['_r'] = d.row_id.str[3:].astype(int); d = d.sort_values('_r')
    same = (d.vehicle_id == d.vehicle_id.shift()) & (d.direction == d.direction.shift()) & (d.stop_seq > d.stop_seq.shift())
    d['gap_before'] = (d.stop_seq - d.stop_seq.shift() - 1).where(same).fillna(0)
    nxt = (d.vehicle_id == d.vehicle_id.shift(-1)) & (d.direction == d.direction.shift(-1)) & (d.stop_seq < d.stop_seq.shift(-1))
    d['gap_after'] = (d.stop_seq.shift(-1) - d.stop_seq - 1).where(nxt).fillna(0)
    return d.drop(columns='_r').loc[df.index]


def add_minute(df, med):
    d = df.copy(); d['_m'] = med; d['_r'] = d.row_id.str[3:].astype(int); d = d.sort_values('_r')
    d['_m'] = d._m.fillna(d._m.median())
    blk = ((d.vehicle_id != d.vehicle_id.shift()) | (d.date != d.date.shift()) | (d.arrive_hour != d.arrive_hour.shift())).cumsum()
    g = d.groupby(blk)._m
    after = g.transform(lambda s: s[::-1].cumsum()[::-1]) - d._m
    before = g.cumsum()
    idx = d.groupby(V + ['arrive_hour']).size().index
    hn = pd.MultiIndex.from_arrays([d.date, d.vehicle_id, d.arrive_hour + 1]).isin(idx)
    hp = pd.MultiIndex.from_arrays([d.date, d.vehicle_id, d.arrive_hour - 1]).isin(idx)
    d['est_min'] = np.clip(np.where(hn, 60 - after / 60, np.where(hp, before / 60, np.nan)), 0, 60)
    d['est_time'] = d.arrive_hour + d.est_min / 60
    return d.drop(columns=['_m', '_r']).loc[df.index]


def seg_med(src, apply):
    md = src.groupby(SEGK).tt_sec.median()
    return apply[SEGK].join(md.rename('x'), on=SEGK).x.values


TR = add_gap(TR)
old = pd.read_parquet('../experiments/oof_final_E41.parquet').set_index('row_id').loc[TR.row_id.values]
y = TR.tt_sec.values; gp = TR.gap_before.values > 0
res = {}
t0 = time.time()
for name, extra in [('A 기록 빠짐', ['gap_before', 'gap_after']), ('B 분 단위 위치', ['est_min', 'est_time'])]:
    o = pd.Series(np.nan, index=TR.index)
    for d in DATES:
        fm = (TR.date != d).values; f, v = TR[fm], TR[~fm]
        if extra[0] == 'est_min':
            mf = np.full(len(f), np.nan)
            for dd in f.date.unique():
                m = (f.date == dd).values; mf[m] = seg_med(f[~m], f[m])
            f, v = add_minute(f, mf), add_minute(v, seg_med(f, v))
        o[v.index] = lgb.LGBMRegressor(**LGB).fit(to_X(f, F + extra), f.tt_sec).predict(to_X(v, F + extra))
    res[name] = o.values
    print(f'[{PH}] {name} 끝 ({time.time() - t0:.0f}초)', flush=True)
for n, p in [('E41 LightGBM', old.lgb.values)] + list(res.items()):
    r = cv_of(pd.Series(p, index=TR.index))
    print(f"[{PH}] {n:14s} 월 {r['월']:.3f} 화 {r['화']:.3f} 수 {r['수']:.3f} 토 {r['토']:.3f} | 예상 {r['예상']:.3f} | 기록 빠진 직후 행 {np.abs(y - p)[gp].mean():.2f} | 나머지 {np.abs(y - p)[~gp].mean():.3f}", flush=True)
