"""E53 평일 누적확률(CDF) 모델: "기준 시간 안에 도착할 확률"을 여러 개 학습해서 구한 중앙값을 E46 목·금 예측과 섞기
  - 기준 시간 [30, 45, 60, 75, 90, 105, 120, 150, 180, 240, 360, 600, 900]초마다 LGBMClassifier (tt_sec <= 기준)
    objective binary, num_leaves 31, min_child_samples 100, learning_rate 0.04, n_estimators 500, random_state 0 (가중치·오버샘플링 없음)
  - 피처·전처리 = E46 평일 전용 모델과 같음 (35개, 회차 구간은 학습 날짜로만 고름)
  - 행마다 확률 13개에 동일 가중치 단조회귀(isotonic, 증가) → 양 끝 (0초, 0), (1800초, 1) 추가 → 확률이 처음 0.5가 되는 시간을 선형 보간 = 새 예측
  - E53 평일 = 0.75 × E46 + 0.25 × 새 예측 (미리 고정). 일요일 = E46 제출값
  final cv      : 월(화·수로 학습)·화(월·수)·수(월·화) → experiments/E53_cv_table.csv, oof_final_E53.parquet
  practice check: 연습 월·화·수로 학습 → 연습 목·금에서 E46과 한 번 비교
  final make    : submissions/final_E53_weekday_cdf.csv
실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E53_weekday_cdf.py final cv
주의: 노트북 cv_feature_experiments.ipynb 의 셀(2,4,6,8,10,27,29,41번)을 실행해서 피처·모델 함수를 가져옴. E46 코드·노트북·제출 파일은 안 건드림.
"""
import json, os, re, sys, time
import numpy as np, pandas as pd, lightgbm as lgb
os.chdir('/Users/kimjuwon/Desktop/에모팀플/bis-travel-time-practice/notebooks')
PH, MODE = sys.argv[1], sys.argv[2]
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
THR = np.array([30, 45, 60, 75, 90, 105, 120, 150, 180, 240, 360, 600, 900], dtype=float)
CLF = dict(objective='binary', num_leaves=31, min_child_samples=100, learning_rate=0.04, n_estimators=500, random_state=0, verbose=-1)
W_E46 = 0.75
dow = lambda df: pd.to_datetime(df.date).dt.day_of_week.values
WD = TRAIN[dow(TRAIN) < 5]
print(f'[{PH}] 학습 평일 날짜: {sorted(WD.date.unique())} | 피처 {len(F)}개', flush=True)


def pav(P):
    """행마다 동일 가중치 단조(증가) 회귀. P: (행, 13)"""
    out = np.empty_like(P)
    for r in range(P.shape[0]):
        v = list(P[r]); w = [1.0] * len(v); idx = [[i] for i in range(len(v))]
        i = 0
        while i < len(v) - 1:
            if v[i] > v[i + 1]:
                nv = (v[i] * w[i] + v[i + 1] * w[i + 1]) / (w[i] + w[i + 1])
                v[i:i + 2] = [nv]; w[i:i + 2] = [w[i] + w[i + 1]]; idx[i:i + 2] = [idx[i] + idx[i + 1]]
                i = max(i - 1, 0)
            else:
                i += 1
        for val, ids in zip(v, idx):
            out[r, ids] = val
    return out


def cdf_median(P):
    Q = pav(P)
    T = np.concatenate([[0.0], THR, [1800.0]])
    Pz = np.hstack([np.zeros((len(Q), 1)), Q, np.ones((len(Q), 1))])
    j = np.argmax(Pz >= 0.5, axis=1)                         # 처음 0.5 이상 (끝이 1이라 항상 있음, j >= 1)
    r = np.arange(len(Q)); p0, p1 = Pz[r, j - 1], Pz[r, j]; t0, t1 = T[j - 1], T[j]
    return t0 + (0.5 - p0) / (p1 - p0) * (t1 - t0)


def fit_pred(fit, apply):
    turn = turn_of(fit)
    f, a = add_turn_hint(fit, turn), add_turn_hint(apply, turn)
    Xf, Xa = to_X(f, F), to_X(a, F)
    P = np.column_stack([lgb.LGBMClassifier(**CLF).fit(Xf, (f.tt_sec <= t).astype(int)).predict_proba(Xa)[:, 1] for t in THR])
    return cdf_median(P), a, P


if MODE == 'cv':
    o41 = pd.read_parquet(f'../experiments/oof_{PH}_E41.parquet').set_index('row_id')
    o44 = pd.read_parquet(f'../experiments/oof_{PH}_E44.parquet').set_index('row_id')
    parts = []; t0 = time.time()
    for d in sorted(WD.date.unique()):
        fit, val = WD[WD.date != d], WD[WD.date == d]
        pc, a, P = fit_pred(fit, val)
        ids = val.row_id.values
        e46 = 0.5 * o41.loc[ids].e41.values + 0.5 * o44.loc[ids].wd_lgb.values
        big = np.abs(o41.loc[ids].cb.values - o44.loc[ids].wd_lgb.values) >= 10
        parts.append(pd.DataFrame({'row_id': ids, 'day': DAY[pd.Timestamp(d).day_of_week], 'y': val.tt_sec.values, 'e46': e46, 'cdf': pc,
                                   'e53': W_E46 * e46 + (1 - W_E46) * pc, 'turn': a.is_turn2.values == 1, 'big_diff': big}))
        print(f'[{PH}] {parts[-1].day.iloc[0]} 끝 (학습 {sorted(fit.date.unique())}, {time.time() - t0:.0f}초) | 단조 위반 행 {(np.diff(P, axis=1) < 0).any(axis=1).mean():.1%}', flush=True)
    R = pd.concat(parts)
    def tab(g):
        e = lambda c: (g.y - g[c]).abs().mean()
        return pd.Series({'행 수': len(g), 'E46 MAE': e('e46'), '새 모델 단독 MAE': e('cdf'), 'E53 MAE': e('e53'), '개선폭(E46−E53)': e('e46') - e('e53')})
    T = pd.concat([R.groupby('day', sort=False).apply(tab), tab(R).to_frame('평일 전체 행').T])
    T.loc['날짜별 단순평균'] = T.iloc[:3].mean(); T.loc['날짜별 단순평균', '행 수'] = np.nan
    S = pd.concat([tab(R[R.big_diff]).to_frame('cb−wd_lgb 차이 ≥ 10초').T, tab(R[~R.big_diff]).to_frame('차이 < 10초').T,
                   tab(R[R.turn]).to_frame('회차 구간').T, tab(R[~R.turn]).to_frame('일반 구간').T])
    print(T.round(3).to_string()); print(S.round(3).to_string(), flush=True)
    pd.concat([T, S]).to_csv('../experiments/E53_cv_table.csv'); R.to_parquet(f'../experiments/oof_{PH}_E53.parquet')
    days = int((T.iloc[:3]['개선폭(E46−E53)'] > 0).sum()); imp = T.loc['평일 전체 행', '개선폭(E46−E53)']
    print(f'[{PH}] 평일 전체 개선폭 {imp:+.3f}초 | 개선 날짜 {days}/3 | 통과 {"예" if imp > 0 and days >= 2 else "아니오"}', flush=True)
else:
    pc, a, P = fit_pred(WD, TEST)
    e46 = pd.read_csv(f'../submissions/{PH}_E46_wd_route_sunmix.csv').set_index('row_id').tt_sec
    TE = TEST.assign(C=pc, E46=e46.loc[TEST.row_id.values].values)
    wd = dow(TE) < 5
    new = np.where(wd, np.round(W_E46 * TE.E46 + (1 - W_E46) * TE.C, 1), TE.E46)
    if MODE == 'check':
        y = TE.tt_sec.values
        for name, p in [('E46', TE.E46.values), ('새 모델 단독', TE.C.values), ('E53', new)]:
            aa = np.abs(y - p)
            print(f'[{PH}] {name:8s} 연습 목·금 {aa[wd].mean():.3f} | 목 {aa[dow(TE) == 3].mean():.3f} | 금 {aa[dow(TE) == 4].mean():.3f}', flush=True)
        print(f'[{PH}] 개선폭 (E46 − E53, 목·금 전체 행) = {np.abs(y - TE.E46.values)[wd].mean() - np.abs(y - new)[wd].mean():+.3f}', flush=True)
    elif MODE == 'make':
        sub = pd.DataFrame({'row_id': TE.row_id.values, 'tt_sec': new}).set_index('row_id').loc[test.row_id].reset_index()
        ss = pd.read_csv(f'../data/{PH}/sample_submission.csv')
        assert len(sub) == len(ss) == 111563 and (sub.row_id.values == ss.row_id.values).all(), 'row_id 순서·행 수'
        assert sub.tt_sec.notna().all() and not sub.row_id.duplicated().any() and (sub.tt_sec >= 0).all(), '결측·중복·음수'
        sid = TE.row_id[~wd]; assert (sub.set_index('row_id').tt_sec.loc[sid].values == e46.loc[sid].values).all(), '일요일 = E46'
        path = f'../submissions/{PH}_E53_weekday_cdf.csv'; assert not os.path.exists(path)
        sub.to_csv(path, index=False)
        print(f'[{PH}] 저장 {path} {sub.shape} | 순서 = sample_submission | 결측·중복·음수 0 | 일요일 {(~wd).sum():,}행 E46과 같음', flush=True)
