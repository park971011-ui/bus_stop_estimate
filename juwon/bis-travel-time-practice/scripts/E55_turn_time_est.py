"""E55 회차 구간에 "추정 회차 시간" 피처 1개 추가 (E41 LightGBM 기준, 대회 train 날짜별 4-fold 검증)
  추정 회차 시간 = 3574 − (같은 날·같은 차량이 같은 시간대(arrive_hour)에 지나간 다른 구간들의 평소 시간 합)
    - 3574초 = train에서 "꽉 찬 1시간 칸" 실제 합의 중앙값 (E48)
    - 평소 시간 = 구간 중앙값. 검증 날짜 행은 학습 날짜로, 학습 날짜 행은 자기 날짜를 뺀 다른 학습 날짜로 계산 (정답 누출 없음)
    - 회차 구간 행이면서 차량의 앞·뒤 시간대 기록이 다 있을 때만 값, 나머지는 결측
  단독으로는 회차 MAE 331.9 (E41 198.5), 상관 0.40이라 피처로만 씀.
실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E55_turn_time_est.py final
결과: 화면 점수 (제출 파일 없음). 약 5분
주의: 노트북 cv_feature_experiments.ipynb 의 셀(2,4,6,8,10,27,29,41번)을 실행해서 피처·모델 함수를 가져옴.
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


old = pd.read_parquet('../experiments/oof_final_E41.parquet').set_index('row_id').loc[TR.row_id.values]
o = pd.Series(np.nan, index=TR.index); t0 = time.time()
for d in DATES:
    fm = (TR.date != d).values
    f, v = TR[fm], TR[~fm]
    mf = np.full(len(f), np.nan)
    for dd in f.date.unique():                  # 학습 행: 자기 날짜 뺀 다른 학습 날짜로
        m = (f.date == dd).values; mf[m] = seg_med(f[~m], f[m])
    f2, v2 = add_est(f, mf), add_est(v, seg_med(f, v))
    o[v.index] = lgb.LGBMRegressor(**LGB).fit(to_X(f2, F + ['turn_est']), f2.tt_sec).predict(to_X(v2, F + ['turn_est']))
    print(f'[{PH}] {DAY[pd.Timestamp(d).day_of_week]} 끝 ({time.time() - t0:.0f}초) | 회차 행 중 값 있음 {v2.turn_est.notna().sum()}/{(v2.is_turn2 == 1).sum()}', flush=True)
y = TR.tt_sec.values; tr = TR.is_turn2.values == 1
for name, p in [('E41 LightGBM', old.lgb.values), ('E55 + 추정 회차 시간', o.values)]:
    r = cv_of(pd.Series(p, index=TR.index))
    print(f"[{PH}] {name:18s} 월 {r['월']:.3f} 화 {r['화']:.3f} 수 {r['수']:.3f} 토 {r['토']:.3f} | 예상 {r['예상']:.3f} | 회차 행 {np.abs(y - p)[tr].mean():.1f} | 나머지 {np.abs(y - p)[~tr].mean():.3f}", flush=True)
