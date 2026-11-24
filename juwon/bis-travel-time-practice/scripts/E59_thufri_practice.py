"""E59 목·금에도 실습용 데이터 쓰기 (실습용.zip 학습 사용은 11/24 확인받음)
  실습 test 목·금(9/26·27)은 대회 test 목·금과 같은 날짜. 피처 = E41 35개 + is_sunday, is_thu, is_fri. LightGBM(E41 설정).
  후보 (미리 고정):
    A  = 대회 train + 실습 train + 실습 test 전부로 학습
    B  = 위에서 평일 행만으로 학습 (E44 평일 전용처럼)
    A½ = E56 목·금 × 0.5 + A × 0.5,  B½ = E56 목·금 × 0.5 + B × 0.5
  검증: 실습 노선 하나 r을 빼고(r의 실습 test 전부 제외, r의 실습 train은 포함) 학습 → r의 실습 test 목·금 예측, 4번.
        비교 기준 = 실습 E56 파일 (지금 최종 E58의 목·금과 같은 방법)
  일요일은 E58 그대로.
실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E59_thufri_practice.py cv   (또는 make A½ 등)
결과: cv → 화면 + experiments/E59_loro.parquet / make → submissions/final_E59_thufri_practice.csv
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
FE = F + ['is_sunday', 'is_thu', 'is_fri']
LGB = {**Gf['PARAMS'], 'min_child_samples': 10, 'learning_rate': 0.02, 'n_estimators': 1500}
DAY, cv_of = Gf['DAY'], Gf['cv_of']
dow = lambda df: pd.to_datetime(df.date).dt.day_of_week.values


def prep(Gx):
    turn = turn_of(Gx['TRAIN'])
    tr, te = add_turn_hint(Gx['TRAIN'], turn), add_turn_hint(Gx['TEST'], turn)
    for d in (tr, te):
        d['is_sunday'] = (dow(d) == 6).astype(int); d['is_thu'] = (dow(d) == 3).astype(int); d['is_fri'] = (dow(d) == 4).astype(int)
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
t0 = time.time()
wdm = lambda df: dow(df) < 5
if MODE == 'cv':
    base = pd.read_csv('../submissions/practice_E56_E46_turnest.csv').set_index('row_id').tt_sec
    out = []
    for r in sorted(PTE.route_nm.unique()):
        m = (PTE.route_nm == r).values
        tr = pd.concat([FTR, PTR, PTE[~m]])
        v = PTE[m & wdm(PTE)]
        pA = fitp(tr).predict(X(v)); pB = fitp(tr[wdm(tr)]).predict(X(v))
        out.append(pd.DataFrame({'row_id': v.row_id.values, 'route': r, 'dow': dow(v), 'y': v.tt_sec.values, 'e56': base.loc[v.row_id].values, 'A': pA, 'B': pB}))
        print(f'[검증] 노선 {r} 끝 ({time.time() - t0:.0f}초)', flush=True)
    L = pd.concat(out); L['A½'] = 0.5 * L.e56 + 0.5 * L.A; L['B½'] = 0.5 * L.e56 + 0.5 * L.B
    L.to_parquet('../experiments/E59_loro.parquet')
    cols = ['e56', 'A', 'B', 'A½', 'B½']
    T = L.groupby('route').apply(lambda g: pd.Series({c: (g.y - g[c]).abs().mean() for c in cols}))
    T.loc['목·금 전체'] = [(L.y - L[c]).abs().mean() for c in cols]
    for k, dd in [('목', 3), ('금', 4)]:
        T.loc[k] = [(L.y - L[c])[L.dow == dd].abs().mean() for c in cols]
    print(T.round(3).rename(columns={'e56': 'E56 (기준)'}).to_string(), flush=True)
    for c in cols[1:]:
        print(f'[검증] {c:3s} − E56 = {T.loc["목·금 전체", c] - T.loc["목·금 전체", "e56"]:+.3f} | 좋아진 노선 {(T.iloc[:4][c] < T.iloc[:4].e56).sum()}/4', flush=True)
else:
    W = sys.argv[2]; assert W in ('A', 'B', 'A½', 'B½')
    ALL = pd.concat([FTR, PTR, PTE])
    m = fitp(ALL if W.startswith('A') else ALL[wdm(ALL)])
    tf = wdm(FTE); p = m.predict(X(FTE[tf]))
    e58 = pd.read_csv('../submissions/final_E58_sunday_practice.csv').set_index('row_id').tt_sec
    e56 = pd.read_csv('../submissions/final_E56_E46_turnest.csv').set_index('row_id').tt_sec
    ids = FTE.row_id.values[tf]
    new = e58.copy()
    new.loc[ids] = np.round(0.5 * e56.loc[ids].values + 0.5 * p, 1) if W.endswith('½') else np.round(p, 1)
    ss = pd.read_csv('../data/final/sample_submission.csv')
    sub = new.loc[ss.row_id].reset_index(); sub.columns = ['row_id', 'tt_sec']
    assert len(sub) == 111563 and (sub.row_id.values == ss.row_id.values).all()
    assert sub.tt_sec.notna().all() and not sub.row_id.duplicated().any() and (sub.tt_sec >= 0).all()
    sids = FTE.row_id.values[~tf]; assert (sub.set_index('row_id').tt_sec.loc[sids].values == e58.loc[sids].values).all(), '일요일 = E58'
    path = '../submissions/final_E59_thufri_practice.csv'; assert not os.path.exists(path); sub.to_csv(path, index=False)
    d = new.loc[ids].values - e58.loc[ids].values
    print(f'저장 {path} {sub.shape} | 순서 = sample_submission | 결측·중복·음수 0 | 일요일 {len(sids):,}행 E58과 같음 | 목·금 E58 대비 평균 {d.mean():+.2f}초, |차이| {np.abs(d).mean():.2f}초', flush=True)
