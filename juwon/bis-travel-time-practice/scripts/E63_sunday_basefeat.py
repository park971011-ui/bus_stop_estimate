"""E63 일요일 모델에 "노선이 바뀌어도 통하는" 기준 시간 피처 4개 추가 (E58 일요일 모델 기준)
  sat_med      : 같은 구간·시간대의 토요일 tt 중앙값 (표본 적으면 구간 토요일 중앙값 쪽으로, K=20)
  wd_med       : 같은 구간·시간대의 월·화·수 tt 중앙값 (같은 방식)
  sat_wd_ratio : sat_med / wd_med
  int_rel      : 그날 배차 / 토요일 배차
  누출 방지: 값을 계산할 행의 날짜는 항상 빼고 계산 (학습 행 = 다른 학습 날짜로, 예측 행 = 학습 행 전체로)
  검증: 실습 노선 하나 r을 빼고(r의 실습 test 제외) 학습 → r의 일요일. 회차 구간 MAE 따로 봄.
실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E63_sunday_basefeat.py cv
"""
import json, os, re, sys, time
import numpy as np, pandas as pd, lightgbm as lgb
os.chdir('/Users/kimjuwon/Desktop/에모팀플/bis-travel-time-practice/notebooks')
nb = json.load(open('cv_feature_experiments.ipynb'))


def load(ph):
    G = {}
    for i in [2, 4, 6, 8, 10, 27, 29, 41]:
        src = ''.join(nb['cells'][i]['source'])
        if i == 2:
            src = re.sub(r"PHASE = '\w+'", f"PHASE = '{ph}'", src, count=1)
        exec(src, G)
    assert G['PHASE'] == ph
    return G


Gp, Gf = load('practice'), load('final')
G = Gf
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


F = Gf['feats_of'](['base', 'seg', 'time', 'trip']) + H
FE = F + ['is_sunday']
LGB = {**Gf['PARAMS'], 'min_child_samples': 10, 'learning_rate': 0.02, 'n_estimators': 1500}
DAY, cv_of = Gf['DAY'], Gf['cv_of']
dow = lambda df: pd.to_datetime(df.date).dt.day_of_week.values


def prep(Gx):
    turn = turn_of(Gx['TRAIN'])
    tr, te = add_turn_hint(Gx['TRAIN'], turn), add_turn_hint(Gx['TEST'], turn)
    for d in (tr, te):
        d['is_sunday'] = (dow(d) == 6).astype(int)
    return tr, te


PTR, PTE = prep(Gp); FTR, FTE = prep(Gf)
CATS = {c: pd.CategoricalDtype(sorted(pd.concat([PTR[c], PTE[c], FTR[c], FTE[c]]).unique())) for c in ['route_id', 'seg_id']}
def X(df):
    x = df[FE].copy()
    for c in CATS: x[c] = x[c].astype(CATS[c])
    return x
fitp = lambda f: lgb.LGBMRegressor(**LGB).fit(X(f), f.tt_sec)
print(f'행 수: 대회 train {len(FTR):,} | 연습 train {len(PTR):,} | 연습 test {len(PTE):,} (일요일 {(PTE.is_sunday == 1).sum():,})', flush=True)
K = 20
NEW = ['sat_med', 'wd_med', 'sat_wd_ratio', 'int_rel']
F2 = FE + NEW


def smooth_med(src, apply):
    g = src.groupby(SEGK + ['arrive_hour']).tt_sec.agg(['median', 'size'])
    s = src.groupby(SEGK).tt_sec.median()
    k = pd.MultiIndex.from_arrays([apply[c] for c in SEGK] + [apply.arrive_hour])
    m, n = g['median'].reindex(k).values, g['size'].reindex(k).fillna(0).values
    sm = s.reindex(pd.MultiIndex.from_arrays([apply[c] for c in SEGK])).values
    return np.where(n > 0, (n * np.nan_to_num(m) + K * sm) / (n + K), sm)


def add_base(src, apply, exclude_own_date):
    a = apply.copy(); a['sat_med'] = np.nan; a['wd_med'] = np.nan
    groups = a.date.unique() if exclude_own_date else [None]
    for dd in groups:
        am = (a.date == dd).values if dd is not None else np.ones(len(a), bool)
        sr = src[src.date != dd] if dd is not None else src
        sd = dow(sr)
        a.loc[am, 'sat_med'] = smooth_med(sr[sd == 5], a[am])
        a.loc[am, 'wd_med'] = smooth_med(sr[sd < 3], a[am])
    a['sat_wd_ratio'] = a.sat_med / a.wd_med
    a['int_rel'] = a.interval / a.interval_sat
    return a


def Xf(df, feats):
    x = df[feats].copy()
    for c in CATS: x[c] = x[c].astype(CATS[c])
    return x


t0 = time.time()
L = pd.read_parquet('../experiments/E60_loro.parquet').set_index('row_id')
Tp = turn_of(Gp['TRAIN']); Tset = set(map(tuple, Tp[SEGK].values))
out = []
for r in sorted(PTE.route_nm.unique()):
    m = (PTE.route_nm == r).values
    tr = pd.concat([FTR, PTR, PTE[~m]]); v = PTE[m & (dow(PTE) == 6)]
    tr2, v2 = add_base(tr, tr, True), add_base(tr, v, False)
    p = lgb.LGBMRegressor(**LGB).fit(Xf(tr2, F2), tr2.tt_sec).predict(Xf(v2, F2))
    out.append(pd.DataFrame({'row_id': v.row_id.values, 'route': r, 'y': v.tt_sec.values, 'e58': L.loc[v.row_id, 'S0 지금 일요일 모델 (E58)'].values,
                             'e60': L.loc[v.row_id, 'S2a + CatBoost 0.2'].values, 'e63': p,
                             'turn': [tuple(x) in Tset for x in v[SEGK].values], 'satnan': v2.sat_med.isna().values}))
    print(f'[검증] 노선 {r} 끝 ({time.time() - t0:.0f}초) | 일요일 행 중 sat_med 결측 {v2.sat_med.isna().mean():.1%}', flush=True)
R = pd.concat(out); R.to_parquet('../experiments/E63_loro.parquet')
cols = ['e58', 'e63', 'e60']
T = R.groupby('route').apply(lambda g: pd.Series({c: (g.y - g[c]).abs().mean() for c in cols}))
T.loc['일요일 전체'] = [(R.y - R[c]).abs().mean() for c in cols]
T.loc['회차 구간'] = [(R.y - R[c])[R.turn].abs().mean() for c in cols]
T.loc['일반 구간'] = [(R.y - R[c])[~R.turn].abs().mean() for c in cols]
print(T.rename(columns={'e58': 'E58 LightGBM', 'e63': 'E63 LightGBM(+4개)', 'e60': '참고 E60 (LGB+CB)'}).round(3).to_string(), flush=True)
print(f'[검증] E63 − E58 = {T.loc["일요일 전체", "e63"] - T.loc["일요일 전체", "e58"]:+.3f} | 좋아진 노선 {(T.iloc[:4].e63 < T.iloc[:4].e58).sum()}/4', flush=True)
