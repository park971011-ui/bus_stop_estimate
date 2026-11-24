"""E58 제출 파일: 목·금 = E56 그대로, 일요일 = E57 모델 (대회 train + 연습 train + 연습 test 전부로 학습, is_sunday 피처).
원래 E57 설명: 연습 데이터(연습 train + 정답 있는 연습 test, 일요일 포함)를 학습에 추가 + is_sunday 피처
  규칙 근거: EVALUATION.md 5. "외부 데이터: 사용하지 않음. 수업에서 제공한 데이터(연습용·대회 데이터)만 씀."
  피처: E41 피처 35개 + is_sunday. 회차 힌트는 각 데이터(연습/대회)의 train으로 회차 구간을 고름.
  모델: E41 LightGBM 설정 (검증은 LightGBM만, 빠르게)

  검증 1 (일요일 옮겨 배우기): 연습 노선 하나 r을 빼고(r의 연습 test 전부 제외, r의 연습 train은 포함 = 대회 노선과 같은 상황)
         대회 train + 연습 train + 다른 노선 연습 test로 학습 → r의 연습 test(목·금·일) 예측. 4번 반복.
         비교: 연습 데이터 E56 제출 방식 파일 (일요일은 토요일처럼·평일처럼 섞기)
  검증 2 (대회 노선 평일·토 손해 없나): 대회 train 날짜별 4-fold, 학습에 연습 데이터 전부 추가 → E41 LightGBM(25.227)과 비교
실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E58_sunday_practice.py
결과: submissions/final_E58_sunday_practice.csv (약 2분)
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
model = fitp(pd.concat([FTR, PTR, PTE]))
sun = dow(FTE) == 6
p_sun = model.predict(X(FTE[sun]))
print(f'학습·예측 {time.time() - t0:.0f}초 | 대회 test 일요일 {sun.sum():,}행', flush=True)
e56 = pd.read_csv('../submissions/final_E56_E46_turnest.csv').set_index('row_id').tt_sec
new = e56.copy()
new.loc[FTE.row_id.values[sun]] = np.round(p_sun, 1)
ss = pd.read_csv('../data/final/sample_submission.csv')
sub = new.loc[ss.row_id].reset_index()
sub.columns = ['row_id', 'tt_sec']
assert len(sub) == len(ss) == 111563 and (sub.row_id.values == ss.row_id.values).all()
assert sub.tt_sec.notna().all() and not sub.row_id.duplicated().any() and (sub.tt_sec >= 0).all()
wd_ids = FTE.row_id.values[~sun]
assert (sub.set_index('row_id').tt_sec.loc[wd_ids].values == e56.loc[wd_ids].values).all(), '목·금 = E56'
path = '../submissions/final_E58_sunday_practice.csv'; assert not os.path.exists(path)
sub.to_csv(path, index=False)
d = (sub.set_index('row_id').tt_sec.loc[FTE.row_id.values[sun]].values - e56.loc[FTE.row_id.values[sun]].values)
print(f'저장 {path} {sub.shape} | 순서 = sample_submission | 결측·중복·음수 0 | 목·금 {len(wd_ids):,}행 E56과 같음', flush=True)
print(f'일요일 E56 대비: 평균 차이 {d.mean():+.2f}초 (음수 = 더 짧게 예측) | 평균 |차이| {np.abs(d).mean():.2f}초', flush=True)
for r in sorted(FTE.route_nm.unique()):
    m = (FTE.route_nm.values[sun] == r); print(f'  노선 {r}: 일요일 평균 차이 {d[m].mean():+.2f}초', flush=True)
