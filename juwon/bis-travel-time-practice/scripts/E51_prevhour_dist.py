"""E51 E46의 평일(목·금) 예측 개선: 평일 전용 LightGBM(E44)에 "직전 시간대 이동" 피처 2개만 추가
  가설: 같은 노선·방향·시간대라도, 이 차량이 직전 시간대에 평소보다 적게 이동했다면 다음 구간 예측에 도움이 되나?
  - prev_n     : 같은 날짜·차량·노선·방향의 정확히 1시간 전(h−1) 기록 행 수 (없으면 결측)
  - prev_ratio : 그 h−1 행들의 dist_m 합 ÷ 같은 노선·방향·(h−1)시의 "차량·날짜별 거리 합" 중앙값
                 중앙값은 학습 날짜에서만 계산. 분모가 없거나 0이면 결측. tt_sec 안 씀. 7시대·전날로 대체 안 함.
  나머지(E41 피처·하이퍼파라미터·seed 0·혼합 E41 0.5 + 평일 전용 0.5·일요일 E46 값)는 그대로.
  ※ 관측 거리는 기록 누락·방향 전환에도 영향을 받으므로 실제 운행 속도로 해석하면 안 됨.

  final cv      : 월(화·수로 학습)·화(월·수)·수(월·화) 검증. 기존 = E41 0.5 + E44 평일 전용 0.5 (저장된 검증 예측)
  practice check: 연습 월·화·수로 학습 → 연습 목·금에서 E46과 한 번 비교
  final make    : 목·금 = 새 예측, 일요일 = E46 → submissions/final_E51_prevhour.csv
실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E51_prevhour_dist.py final cv
주의: 노트북 cv_feature_experiments.ipynb 의 셀(2,4,6,8,10,27,29,41번)을 실행해서 피처·모델 함수를 가져옴. E44·E46 파일은 안 건드림.
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
P2 = ['prev_n', 'prev_ratio']
LGB = {**G['PARAMS'], 'min_child_samples': 10, 'learning_rate': 0.02, 'n_estimators': 1500}
dow = lambda df: pd.to_datetime(df.date).dt.day_of_week.values
WD = TRAIN[dow(TRAIN) < 5]
VK = ['date', 'vehicle_id', 'route_id', 'direction']
print(f'[{PH}] 학습 평일 날짜: {sorted(WD.date.unique())}', flush=True)


def add_prev(df, fit):
    """df 자기 날짜의 h−1 기록으로 피처 계산. 중앙값(분모)은 fit(학습 날짜)에서만."""
    df = df.copy()
    agg = df.groupby(VK + ['arrive_hour']).dist_m.agg(['size', 'sum'])
    k = pd.MultiIndex.from_arrays([df[c] for c in VK] + [df.arrive_hour - 1])
    prev = agg.reindex(k)
    df['prev_n'] = prev['size'].values                                   # 없으면 NaN
    vs = fit.groupby(VK + ['arrive_hour']).dist_m.sum()                   # 차량·날짜별 시간대 거리 합 (학습 날짜)
    med = vs.groupby(level=['route_id', 'direction', 'arrive_hour']).median()
    km = pd.MultiIndex.from_arrays([df.route_id, df.direction, df.arrive_hour - 1])
    den = med.reindex(km).values
    den = np.where((den > 0), den, np.nan)
    df['prev_ratio'] = prev['sum'].values / den
    return df


def fit_pred(fit, apply, extra):
    turn = turn_of(fit)
    f, a = add_turn_hint(fit, turn), add_turn_hint(apply, turn)
    feats = F
    if extra:
        f, a = add_prev(f, fit), add_prev(a, fit); feats = F + P2
    return lgb.LGBMRegressor(**LGB).fit(to_X(f, feats), f.tt_sec).predict(to_X(a, feats)), turn, a


if MODE == 'cv':
    old41 = pd.read_parquet(f'../experiments/oof_{PH}_E41.parquet').set_index('row_id')
    old44 = pd.read_parquet(f'../experiments/oof_{PH}_E44.parquet').set_index('row_id')
    rows, all_old, all_new, all_y, all_has, all_turn = [], [], [], [], [], []
    t0 = time.time()
    for d in sorted(WD.date.unique()):
        fit, val = WD[WD.date != d], WD[WD.date == d]
        pB2, turn, a = fit_pred(fit, val, True)
        ids = val.row_id.values; y = val.tt_sec.values
        old = 0.5 * old41.loc[ids].e41.values + 0.5 * old44.loc[ids].wd_lgb.values
        new = 0.5 * old41.loc[ids].e41.values + 0.5 * pB2
        has = ~np.isnan(a.prev_n.values); tr = a.is_turn2.values == 1
        name = DAY[pd.Timestamp(d).day_of_week]
        ao, an = np.abs(y - old), np.abs(y - new)
        rows.append({'날짜': name, '행 수': len(val), '기존 MAE': ao.mean(), '새 MAE': an.mean(), '개선폭(기존−새)': ao.mean() - an.mean(),
                     '직전 기록 있음 %': has.mean() * 100, '일반 구간 개선폭': ao[~tr].mean() - an[~tr].mean(), '회차 구간 개선폭': ao[tr].mean() - an[tr].mean(),
                     '직전 기록 있는 행 개선폭': ao[has].mean() - an[has].mean()})
        for L, v in [(all_old, old), (all_new, new), (all_y, y), (all_has, has), (all_turn, tr)]: L.append(v)
        print(f'[{PH}] {name} 끝 (학습 {sorted(fit.date.unique())}, {time.time() - t0:.0f}초)', flush=True)
    r = pd.DataFrame(rows).set_index('날짜')
    o, n, y, has, tr = (np.concatenate(L) for L in (all_old, all_new, all_y, all_has, all_turn))
    ao, an = np.abs(y - o), np.abs(y - n)
    r.loc['세 날짜 평균'] = r.mean(); r.loc['세 날짜 평균', '행 수'] = np.nan
    r.loc['전체 평일 행'] = [len(y), ao.mean(), an.mean(), ao.mean() - an.mean(), has.mean() * 100,
                         ao[~tr].mean() - an[~tr].mean(), ao[tr].mean() - an[tr].mean(), ao[has].mean() - an[has].mean()]
    print(r.round(3).to_string(), flush=True)
    print(f'[{PH}] 회차 구간 행 수 {tr.sum():,} ({tr.mean():.2%}) | 일반 구간 행 수 {(~tr).sum():,}', flush=True)
    imp = r.loc['세 날짜 평균', '개선폭(기존−새)']; days = int((r.drop(['세 날짜 평균', '전체 평일 행'])['개선폭(기존−새)'] > 0).sum())
    share = (dow(TEST) < 5).mean()
    print(f'[{PH}] 기준(E46 평일) 세 날짜 평균 {r.loc["세 날짜 평균", "기존 MAE"]:.3f} (기존 기록 23.261) | 개선 날짜 {days}/3', flush=True)
    print(f'[{PH}] 전체 MAE 효과 추정: {imp:+.3f} × 대회 test 평일 비율 {share:.4f} = {imp * share:+.3f}초 (검증 기반 추정치), 목표까지 0.277초 중 {imp * share / 0.277:.1%}', flush=True)
    r.to_csv(f'../experiments/E51_cv_table.csv')
else:
    pB2, turn, a = fit_pred(WD, TEST, True)
    e41 = pd.read_csv(f'../submissions/{PH}_E41_turnhint_sunmix.csv').set_index('row_id').tt_sec
    e46 = pd.read_csv(f'../submissions/{PH}_E46_wd_route_sunmix.csv').set_index('row_id').tt_sec
    TE = TEST.assign(B2=pB2, E41=e41.loc[TEST.row_id.values].values, E46=e46.loc[TEST.row_id.values].values)
    wd = dow(TE) < 5
    new = np.where(wd, np.round(0.5 * TE.E41 + 0.5 * TE.B2, 1), TE.E46)
    print(f'[{PH}] 직전 기록 있는 test 평일 행 {(~np.isnan(a.prev_n.values))[wd].mean():.1%}', flush=True)
    if MODE == 'check':
        y = TE.tt_sec.values
        for name, p in [('E46', TE.E46.values), ('E51', new)]:
            aa = np.abs(y - p)
            print(f'[{PH}] {name} 연습 목·금 {aa[wd].mean():.3f} | 목 {aa[dow(TE) == 3].mean():.3f} | 금 {aa[dow(TE) == 4].mean():.3f}', flush=True)
        print(f'[{PH}] E51 − E46 (목·금) = {np.abs(y - new)[wd].mean() - np.abs(y - TE.E46.values)[wd].mean():+.3f}', flush=True)
    elif MODE == 'make':
        sub = pd.DataFrame({'row_id': TE.row_id.values, 'tt_sec': new}).set_index('row_id').loc[test.row_id].reset_index()
        ss = pd.read_csv(f'../data/{PH}/sample_submission.csv')
        assert len(sub) == len(ss) and (sub.row_id.values == ss.row_id.values).all(), 'row_id 순서·행 수'
        assert sub.tt_sec.notna().all() and not sub.row_id.duplicated().any() and (sub.tt_sec >= 0).all(), '결측·중복·음수'
        sid = TE.row_id[~wd]
        assert (sub.set_index('row_id').tt_sec.loc[sid].values == e46.loc[sid].values).all(), '일요일 = E46'
        path = f'../submissions/{PH}_E51_prevhour.csv'; assert not os.path.exists(path)
        sub.to_csv(path, index=False)
        print(f'[{PH}] 저장 {path} {sub.shape} | row_id 순서·행 수 = sample_submission | 결측·중복·음수 0 | 일요일 {(~wd).sum():,}행 E46과 완전히 같음', flush=True)
