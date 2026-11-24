"""E57 연습 데이터(연습 train + 정답 있는 연습 test, 일요일 포함)를 학습에 추가 + is_sunday 피처
  규칙 근거: EVALUATION.md 5. "외부 데이터: 사용하지 않음. 수업에서 제공한 데이터(연습용·대회 데이터)만 씀."
  피처: E41 피처 35개 + is_sunday. 회차 힌트는 각 데이터(연습/대회)의 train으로 회차 구간을 고름.
  모델: E41 LightGBM 설정 (검증은 LightGBM만, 빠르게)

  검증 1 (일요일 옮겨 배우기): 연습 노선 하나 r을 빼고(r의 연습 test 전부 제외, r의 연습 train은 포함 = 대회 노선과 같은 상황)
         대회 train + 연습 train + 다른 노선 연습 test로 학습 → r의 연습 test(목·금·일) 예측. 4번 반복.
         비교: 연습 데이터 E56 제출 방식 파일 (일요일은 토요일처럼·평일처럼 섞기)
  검증 2 (대회 노선 평일·토 손해 없나): 대회 train 날짜별 4-fold, 학습에 연습 데이터 전부 추가 → E41 LightGBM(25.227)과 비교
실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E57_add_practice.py
결과: 화면 점수 + experiments/E57_loro.parquet, oof_final_E57.parquet (약 30분)
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
t0 = time.time()

# 검증 1
base = pd.read_csv('../submissions/practice_E56_E46_turnest.csv').set_index('row_id').tt_sec
e46 = pd.read_csv('../submissions/practice_E46_wd_route_sunmix.csv').set_index('row_id').tt_sec
out = []
for r in sorted(PTE.route_nm.unique()):
    m = (PTE.route_nm == r).values
    model = fitp(pd.concat([FTR, PTR, PTE[~m]]))
    v = PTE[m]
    out.append(pd.DataFrame({'row_id': v.row_id.values, 'route': r, 'dow': dow(v), 'y': v.tt_sec.values, 'e57': model.predict(X(v)),
                             'e56': base.loc[v.row_id].values, 'e46': e46.loc[v.row_id].values}))
    print(f'[검증1] 노선 {r} 끝 ({time.time() - t0:.0f}초)', flush=True)
L = pd.concat(out); L.to_parquet('../experiments/E57_loro.parquet')
for k, g in [('일', L.dow == 6), ('목·금', L.dow < 5), ('전체', L.dow >= 0)]:
    x = L[g]; e = lambda c: (x.y - x[c]).abs().mean()
    print(f'[검증1] {k:4s} {len(x):6,}행: E46 {e("e46"):.3f} | E56 {e("e56"):.3f} | E57(빠진 노선) {e("e57"):.3f} | E57 − E56 {e("e57") - e("e56"):+.3f}', flush=True)
print(L[L.dow == 6].groupby('route').apply(lambda x: pd.Series({'E56 일': (x.y - x.e56).abs().mean(), 'E57 일': (x.y - x.e57).abs().mean()})).round(3).to_string(), flush=True)
x = L[L.dow == 6]
for a in [0.5]:
    print(f'[검증1] 일요일 E56 × 0.5 + E57 × 0.5 (참고): {(x.y - (0.5 * x.e56 + 0.5 * x.e57)).abs().mean():.3f}', flush=True)

# 검증 2
o41 = pd.read_parquet('../experiments/oof_final_E41.parquet').set_index('row_id').loc[FTR.row_id.values]
o = pd.Series(np.nan, index=FTR.index)
for d in sorted(FTR.date.unique()):
    m = (FTR.date == d).values
    o[FTR.index[m]] = fitp(pd.concat([FTR[~m], PTR, PTE])).predict(X(FTR[m]))
    print(f'[검증2] {DAY[pd.Timestamp(d).day_of_week]} 끝 ({time.time() - t0:.0f}초)', flush=True)
for n, p in [('E41 LightGBM', o41.lgb.values), ('E57 LightGBM (+연습 데이터)', o.values)]:
    r = cv_of(pd.Series(p, index=FTR.index))
    print(f"[검증2] {n:24s} 월 {r['월']:.3f} 화 {r['화']:.3f} 수 {r['수']:.3f} 토 {r['토']:.3f} | 예상 {r['예상']:.3f}", flush=True)
pd.DataFrame({'row_id': FTR.row_id.values, 'e57_lgb': o.values}).to_parquet('../experiments/oof_final_E57.parquet')
print(f'끝 ({time.time() - t0:.0f}초)')
