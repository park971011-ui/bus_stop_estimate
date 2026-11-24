"""E60 일요일 모델 다듬기 (E58 일요일 모델 = 대회 train + 실습 train + 실습 test로 학습한 LightGBM, is_sunday)
  규칙(미리 고정):
    1단계: 추정 회차 시간(turn_est, E56) 피처를 넣은 LightGBM vs 지금 일요일 모델(E58)
    2단계: 1단계에서 더 좋은 피처로 (a) LightGBM 0.8 + CatBoost 0.2 (b) LightGBM seed 0~4 평균 을 각각 시험
    3단계: 좋아진 것만 합친 최종 조합을 한 번 확인
  검증: 실습 노선 하나 r을 빼고(r의 실습 test 전부 제외, r의 실습 train은 포함) 학습 → r의 실습 test 일요일 예측, 4번.
실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E60_sunday_refine.py cv   (또는 make)
결과: cv → 화면 + experiments/E60_loro.parquet, E60_choice.json / make → submissions/final_E60_sunday_refine.csv
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
MODE = sys.argv[1]
V = ['date', 'vehicle_id']; C = 3574.0
CB = dict(loss_function='MAE', iterations=2000, learning_rate=0.08, depth=10, l2_leaf_reg=3, random_seed=0, verbose=0, thread_count=-1)
F0 = FE; F1 = FE + ['turn_est']


def with_est(train, apply):
    """학습 행: 자기 날짜 뺀 다른 날짜로 구간 중앙값 / 예측 행: 학습 행 전체로"""
    mf = np.full(len(train), np.nan)
    for dd in train.date.unique():
        m = (train.date == dd).values; mf[m] = seg_med(train[~m], train[m])
    return add_est(train, mf), add_est(apply, seg_med(train, apply))


def Xf(df, feats):
    x = df[feats].copy()
    for c in CATS: x[c] = x[c].astype(CATS[c])
    return x


def run(tr, ap, feats, cfg):
    """cfg: seeds, cb"""
    P = [lgb.LGBMRegressor(**{**LGB, 'random_state': s}).fit(Xf(tr, feats), tr.tt_sec).predict(Xf(ap, feats)) for s in range(cfg['seeds'])]
    p = np.mean(P, axis=0)
    if cfg['cb']:
        cb = CatBoostRegressor(**CB, cat_features=['route_id', 'seg_id'])
        x = tr[feats].copy(); xa = ap[feats].copy()
        p = 0.8 * p + 0.2 * cb.fit(x, tr.tt_sec).predict(xa)
    return p


t0 = time.time()
if MODE == 'cv':
    sets = {}
    for r in sorted(PTE.route_nm.unique()):
        m = (PTE.route_nm == r).values
        tr = pd.concat([FTR, PTR, PTE[~m]]); v = PTE[m & (dow(PTE) == 6)]
        sets[r] = (tr, v) + with_est(tr, v)
    L = pd.concat([pd.DataFrame({'row_id': v.row_id.values, 'route': r, 'y': v.tt_sec.values}) for r, (tr, v, _, _) in sets.items()])
    def ev(name, fn):
        L[name] = np.concatenate([fn(*sets[r]) for r in sets])
        e = L.groupby('route').apply(lambda g: (g.y - g[name]).abs().mean())
        print(f'[검증] {name:28s} 일요일 {(L.y - L[name]).abs().mean():.3f} | ' + ' '.join(f'{k} {x:.3f}' for k, x in e.items()) + f'  ({time.time() - t0:.0f}초)', flush=True)
        return (L.y - L[name]).abs().mean()
    s0 = ev('S0 지금 일요일 모델 (E58)', lambda tr, v, tre, ve: run(tr, v, F0, {'seeds': 1, 'cb': False}))
    s1 = ev('S1 + 추정 회차 시간', lambda tr, v, tre, ve: run(tre, ve, F1, {'seeds': 1, 'cb': False}))
    use_est = s1 < s0; feats = F1 if use_est else F0; base = min(s0, s1)
    print(f'[검증] → 1단계: 추정 회차 시간 {"넣음" if use_est else "안 넣음"}', flush=True)
    pick = (lambda tr, v, tre, ve: (tre, ve)) if use_est else (lambda tr, v, tre, ve: (tr, v))
    sa = ev('S2a + CatBoost 0.2', lambda *a: run(*pick(*a), feats, {'seeds': 1, 'cb': True}))
    sb = ev('S2b seed 5개 평균', lambda *a: run(*pick(*a), feats, {'seeds': 5, 'cb': False}))
    cfg = {'est': bool(use_est), 'cb': bool(sa < base), 'seeds': 5 if sb < base else 1}
    print(f'[검증] → 2단계 채택: CatBoost {"O" if cfg["cb"] else "X"}, seed 5개 {"O" if cfg["seeds"] == 5 else "X"}', flush=True)
    if cfg['cb'] and cfg['seeds'] == 5:
        ev('S3 최종 조합', lambda *a: run(*pick(*a), feats, cfg))
    L.to_parquet('../experiments/E60_loro.parquet')
    json.dump(cfg, open('../experiments/E60_choice.json', 'w'))
    print(f'[검증] 최종 조합 {cfg} (E58 일요일 {s0:.3f})', flush=True)
else:
    cfg = json.load(open('../experiments/E60_choice.json')); print('조합', cfg, flush=True)
    ALL = pd.concat([FTR, PTR, PTE]); sun = dow(FTE) == 6; ap = FTE[sun]
    if cfg['est']:
        ALL, ap = with_est(ALL, ap)
    p = run(ALL, ap, F1 if cfg['est'] else F0, cfg)
    e58 = pd.read_csv('../submissions/final_E58_sunday_practice.csv').set_index('row_id').tt_sec
    new = e58.copy(); ids = FTE.row_id.values[sun]; new.loc[ids] = np.round(p, 1)
    ss = pd.read_csv('../data/final/sample_submission.csv')
    sub = new.loc[ss.row_id].reset_index(); sub.columns = ['row_id', 'tt_sec']
    assert len(sub) == 111563 and (sub.row_id.values == ss.row_id.values).all()
    assert sub.tt_sec.notna().all() and not sub.row_id.duplicated().any() and (sub.tt_sec >= 0).all()
    wid = FTE.row_id.values[~sun]; assert (sub.set_index('row_id').tt_sec.loc[wid].values == e58.loc[wid].values).all(), '목·금 = E58'
    path = '../submissions/final_E60_sunday_refine.csv'; assert not os.path.exists(path); sub.to_csv(path, index=False)
    d = new.loc[ids].values - e58.loc[ids].values
    print(f'저장 {path} {sub.shape} | 순서 = sample_submission | 결측·중복·음수 0 | 목·금 {len(wid):,}행 E58과 같음 | 일요일 E58 대비 평균 {d.mean():+.2f}초, |차이| {np.abs(d).mean():.2f}초 ({time.time() - t0:.0f}초)', flush=True)
