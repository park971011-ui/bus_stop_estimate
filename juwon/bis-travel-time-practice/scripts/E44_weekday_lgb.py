"""E44 평일 전용 LightGBM: 목·금 예측을 월·화·수만으로 학습한 모델로 만들면 E41보다 좋은가?

E41과 같은 피처(E5 25개 + 회차 힌트 10개)·MAE 손실·하이퍼파라미터·seed 0. 학습 날짜만 평일로 제한.
회차 구간(상행 중앙값 최대 구간)은 매번 그 학습 날짜 안에서만 다시 고름.

  final cv        : 대회 train, 월(화·수로 학습)·화(월·수)·수(월·화) 검증 → A·B·C·D 비교, experiments/oof_final_E44.parquet
  practice check W: 연습 데이터 월·화·수로 학습 → 연습 목·금에서 E41과 한 번 비교 (W = 후보 B 또는 D)
  final make W    : 대회 월·화·수로 학습 → 목·금은 후보 예측, 일요일은 E41 그대로 → submissions/final_E44_weekday_lgb.csv
    A = E41 LightGBM 단독 / B = 평일 전용 LightGBM 단독 / C = E41 최종 / D = C × 0.5 + B × 0.5
실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E44_weekday_lgb.py final cv
주의: 노트북 cv_feature_experiments.ipynb 의 셀(2,4,6,8,10,27,29,41번)을 실행해서 피처·모델 함수를 가져옴.
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
LGB = {**G['PARAMS'], 'min_child_samples': 10, 'learning_rate': 0.02, 'n_estimators': 1500}
dow = lambda df: pd.to_datetime(df.date).dt.day_of_week.values
WD = TRAIN[dow(TRAIN) < 5]                       # 학습에 쓸 평일 (월·화·수)
print(f'[{PH}] 학습 평일 날짜: {sorted(WD.date.unique())}', flush=True)


def fit_pred(fit, apply):
    turn = turn_of(fit)
    f, a = add_turn_hint(fit, turn), add_turn_hint(apply, turn)
    return lgb.LGBMRegressor(**LGB).fit(to_X(f, F), f.tt_sec).predict(to_X(a, F)), turn


if MODE == 'cv':
    old = pd.read_parquet(f'../experiments/oof_{PH}_E41.parquet').set_index('row_id')
    rows, oof = [], []
    t0 = time.time()
    for d in sorted(WD.date.unique()):
        fit, val = WD[WD.date != d], WD[WD.date == d]
        pB, turn = fit_pred(fit, val)
        o = old.loc[val.row_id.values]; y = val.tt_sec.values
        P = {'A': o.lgb.values, 'B': pB, 'C': o.e41.values, 'D': 0.5 * o.e41.values + 0.5 * pB}
        name = DAY[pd.Timestamp(d).day_of_week]
        rows.append({'날짜': name, **{k: float(np.abs(y - p).mean()) for k, p in P.items()}})
        oof.append(pd.DataFrame({'row_id': val.row_id.values, 'wd_lgb': pB}))
        print(f'[{PH}] {name} 끝 (학습 날짜 {sorted(fit.date.unique())}, 회차 구간 {len(turn)}개, {time.time() - t0:.0f}초)', flush=True)
    r = pd.DataFrame(rows).set_index('날짜'); r.loc['평균'] = r.mean()
    r['B-A'] = r.B - r.A; r['B-C'] = r.B - r.C; r['D-C'] = r.D - r.C
    print(r.round(3).to_string(), flush=True)
    pd.concat(oof).to_parquet(f'../experiments/oof_{PH}_E44.parquet')
    share = (dow(TEST) < 5).mean()
    for k in ['B', 'D']:
        dm = r.loc['평균', f'{k}-C']; better = int((r.drop('평균')[f'{k}-C'] < 0).sum())
        print(f'[{PH}] 후보 {k}: 평일 평균 E41 최종 대비 {dm:+.3f} | 개선된 날짜 {better}/3 | 전체 MAE 효과 추정 {dm:+.3f} × 목·금 비율 {share:.4f} = {dm * share:+.3f} (검증 기반 추정치)', flush=True)
else:
    W = sys.argv[3]; assert W in ('B', 'D')
    pB, turn = fit_pred(WD, TEST)
    e41 = pd.read_csv(f'../submissions/{PH}_E41_turnhint_sunmix.csv').set_index('row_id').tt_sec
    TE = TEST.assign(B=pB, E41=e41.loc[TEST.row_id.values].values)
    wd = dow(TE) < 5
    new = np.where(wd, TE.B if W == 'B' else 0.5 * TE.E41 + 0.5 * TE.B, TE.E41)
    if MODE == 'check':
        assert 'tt_sec' in test.columns
        y = TE.tt_sec.values
        for name, p in [('E41', TE.E41.values), (f'후보 {W}', new)]:
            a = np.abs(y - p)
            print(f'[{PH}] {name:6s} 연습 목·금 {a[wd].mean():.3f} | 목 {a[dow(TE) == 3].mean():.3f} | 금 {a[dow(TE) == 4].mean():.3f}', flush=True)
        print(f'[{PH}] 후보 {W} - E41 (목·금) = {np.abs(y - new)[wd].mean() - np.abs(y - TE.E41.values)[wd].mean():+.3f}', flush=True)
    elif MODE == 'make':
        sub = pd.DataFrame({'row_id': TE.row_id.values, 'tt_sec': np.round(new, 1)}).set_index('row_id').loc[test.row_id].reset_index()
        if os.path.exists(f'../data/{PH}/sample_submission.csv'):           # 연습 데이터에는 없음
            ss = pd.read_csv(f'../data/{PH}/sample_submission.csv')
            assert list(sub.columns) == list(ss.columns) and (sub.row_id.values == ss.row_id.values).all(), 'row_id 순서'
        assert sub.tt_sec.notna().all() and not sub.row_id.duplicated().any(), '결측·중복'
        print(f'[{PH}] 음수 예측 {(sub.tt_sec < 0).sum()}개', flush=True)
        assert PH != 'final' or (sub.tt_sec >= 0).all(), '대회 파일에 음수'
        sun_ids = TE.row_id[~wd]
        assert (sub.set_index('row_id').tt_sec.loc[sun_ids].values == e41.loc[sun_ids].values).all(), '일요일 = E41'
        path = f'../submissions/{PH}_E44_weekday_lgb.csv'
        assert not os.path.exists(path), '이미 있음'
        sub.to_csv(path, index=False)
        print(f'[{PH}] 저장 {path} {sub.shape} | 결측 0 중복 0 (대회 파일은 row_id 순서·음수 0도 확인) | 일요일 {(~wd).sum():,}행 E41과 완전히 같음', flush=True)
        print(f'[{PH}] 목·금 예측 E41 대비 평균 |차이| {np.abs(new - TE.E41.values)[wd].mean():.2f}초 | 최소 {sub.tt_sec.min()}', flush=True)
