"""E62 일요일에 "주말 전용" LightGBM 섞기 (목·금에서 평일 전용 모델 E44가 통한 것처럼)
  주말 전용 = E58과 같은 피처·설정, 대회 train 토 + 실습 train 토 + 실습 test 일 행만으로 학습
  일요일 예측 = E60 일요일 예측 × 0.5 + 주말 전용 × 0.5 (미리 고정). 목·금은 E60 그대로.
  검증: 실습 노선 하나 r을 빼고(r의 실습 test 제외) 학습 → r의 일요일. E60 일요일 검증 예측은 experiments/E60_loro.parquet 재사용.
실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E62_weekend_model.py cv   (또는 make)
결과: cv → 화면 + experiments/E62_loro.parquet / make → submissions/final_E62_weekend_model.csv
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
MODE = sys.argv[1]
wk = lambda df: dow(df) >= 5
t0 = time.time()
if MODE == 'cv':
    L = pd.read_parquet('../experiments/E60_loro.parquet').set_index('row_id')
    out = []
    for r in sorted(PTE.route_nm.unique()):
        m = (PTE.route_nm == r).values
        tr = pd.concat([FTR, PTR, PTE[~m]]); tr = tr[wk(tr)]
        v = PTE[m & (dow(PTE) == 6)]
        out.append(pd.DataFrame({'row_id': v.row_id.values, 'route': r, 'y': v.tt_sec.values, 'e60': L.loc[v.row_id, 'S2a + CatBoost 0.2'].values,
                                 'wk': fitp(tr).predict(X(v))}))
        print(f'[검증] 노선 {r} 끝 (주말 학습 행 {len(tr):,}, {time.time() - t0:.0f}초)', flush=True)
    R = pd.concat(out); R['mix'] = 0.5 * R.e60 + 0.5 * R.wk; R.to_parquet('../experiments/E62_loro.parquet')
    T = R.groupby('route').apply(lambda g: pd.Series({c: (g.y - g[c]).abs().mean() for c in ['e60', 'wk', 'mix']}))
    T.loc['일요일 전체'] = [(R.y - R[c]).abs().mean() for c in ['e60', 'wk', 'mix']]
    print(T.rename(columns={'e60': 'E60', 'wk': '주말 전용', 'mix': 'E60 반 + 주말 반'}).round(3).to_string(), flush=True)
    print(f'[검증] 반반 − E60 = {T.loc["일요일 전체", "mix"] - T.loc["일요일 전체", "e60"]:+.3f} | 좋아진 노선 {(T.iloc[:4].mix < T.iloc[:4].e60).sum()}/4', flush=True)
else:
    ALL = pd.concat([FTR, PTR, PTE]); sun = dow(FTE) == 6
    pw = fitp(ALL[wk(ALL)]).predict(X(FTE[sun]))
    e60 = pd.read_csv('../submissions/final_E60_sunday_refine.csv').set_index('row_id').tt_sec
    ids = FTE.row_id.values[sun]; new = e60.copy(); new.loc[ids] = np.round(0.5 * e60.loc[ids].values + 0.5 * pw, 1)
    ss = pd.read_csv('../data/final/sample_submission.csv')
    sub = new.loc[ss.row_id].reset_index(); sub.columns = ['row_id', 'tt_sec']
    assert len(sub) == 111563 and (sub.row_id.values == ss.row_id.values).all()
    assert sub.tt_sec.notna().all() and not sub.row_id.duplicated().any() and (sub.tt_sec >= 0).all()
    wid = FTE.row_id.values[~sun]; assert (sub.set_index('row_id').tt_sec.loc[wid].values == e60.loc[wid].values).all()
    path = '../submissions/final_E62_weekend_model.csv'; assert not os.path.exists(path); sub.to_csv(path, index=False)
    d = new.loc[ids].values - e60.loc[ids].values
    print(f'저장 {path} {sub.shape} | 순서 = sample_submission | 결측·중복·음수 0 | 목·금 E60과 같음 | 일요일 E60 대비 평균 {d.mean():+.2f}초, |차이| {np.abs(d).mean():.2f}초', flush=True)
