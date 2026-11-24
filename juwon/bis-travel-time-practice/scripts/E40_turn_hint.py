"""E40 회차 구간(종점 대기)에 "얼마나 쉬었는지" 간접 힌트 주기 — 대회 train 검증 비교 (LightGBM E30 설정만):
  A) 기준 E30 (E5 피처 25개)
  B) E5 + 회차 힌트 피처 (회차 구간 행에만 값, 나머지 행은 비움)
  S) A 예측에서 회차 구간 행만 "회차 전용 작은 모델" 예측으로 바꿈

회차 구간: 노선마다 상행(direction 0) 구간 중 train 중앙값이 가장 긴 구간 (E34 새 규칙과 같음)
회차 힌트 (입력 arrive_hour만 사용, 정답 안 씀):
  - 같은 운행에서 바로 앞 구간과의 arrive_hour 차이, 같은 시간 칸의 구간 수·거리 합, 칸 안 앞·뒤 구간 수
  - 차량 하루 전체에서 그 시·앞 시·뒤 시의 행 수
  - 다음 운행(반대 방향)이 몇 시간 뒤 시작하는지, 그 첫 시간 칸의 행 수
  (회차 구간은 상행 마지막 또는 끝에서 둘째 구간이라 바로 다음 행은 반대 방향 다음 운행에 있음)
  → 쉬는 시간이 길면 시간이 넘어가거나 같은 칸에 든 구간 수가 줄어드는 흔적이 남음

실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E40_turn_hint.py final   (또는 practice)
결과: 화면에 검증 점수만 (제출 파일 없음). 약 15~20분
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
TRAIN, TEST = G['TRAIN'], G['TEST']
assert G['PHASE'] == PH

# 회차 구간: 노선마다 상행 구간 중 train 중앙값 최대
SEGK = ['route_id', 'direction', 'from_stop', 'to_stop']
med = TRAIN[TRAIN.direction == 0].groupby(SEGK).tt_sec.median().reset_index()
TURN = med.loc[med.groupby('route_id').tt_sec.idxmax(), SEGK].assign(is_turn2=1)

TRIP = ['date', 'vehicle_id', 'route_id', 'direction', 'trip_no']
VD = ['date', 'vehicle_id']          # 차량 하루 (확인: 차량 하루에 노선은 항상 1개)
H = ['t_prev_gap', 't_blk_n', 't_blk_dist', 't_blk_before', 't_blk_after',
     'v_blk_n', 'v_prev_n', 'v_next_n', 'v_next_trip_gap', 'v_next_trip_n0']


def add_turn_hint(df):
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


TR = add_turn_hint(TRAIN)
TE = add_turn_hint(TEST)
G['TRAIN'] = TR
LGB = {**G['PARAMS'], 'min_child_samples': 10, 'learning_rate': 0.02, 'n_estimators': 1500}
E5 = G['feats_of'](['base', 'seg', 'time', 'trip'])
DATES, DAY, summarize, mae, to_X = G['DATES'], G['DAY'], G['summarize'], G['mae'], G['to_X']
tr_t, te_t = TR[TR.is_turn2 == 1], TE[TE.is_turn2 == 1]
print(f'[{PH}] 회차 구간 행: train {len(tr_t):,} ({len(tr_t) / len(TR):.2%}) | test {len(te_t):,} ({len(te_t) / len(TE):.2%})', flush=True)
print(f'[{PH}] 회차 행 중 다음 운행 찾음: train {tr_t.v_next_trip_gap.notna().mean():.1%} | test {te_t.v_next_trip_gap.notna().mean():.1%}', flush=True)
for gap in [0, 1, 2]:
    m = tr_t.v_next_trip_gap == gap
    print(f'[{PH}] train 회차 행: 다음 운행이 {gap}시간 뒤 시작 {m.mean():5.1%} → tt 중앙값 {tr_t.tt_sec[m].median():.0f}초', flush=True)

SMALL = dict(objective='regression_l1', learning_rate=0.03, num_leaves=15, min_child_samples=20, n_estimators=400,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)
SF = ['route_id', 'arrive_hour', 'is_weekend', 'interval', 'trip_no', 'hours_in_service'] + H


def oof(fn):
    out = pd.Series(np.nan, index=TR.index)
    for d in DATES:
        f, v = TR[TR.date != d], TR[TR.date == d]
        out[v.index] = fn(f, v)
    return out


def report(name, p, t0):
    sc = summarize({DAY[pd.Timestamp(d).day_of_week]: mae(TR.tt_sec[TR.date == d], p[TR.date == d]) for d in DATES})
    t = TR.is_turn2 == 1
    print(f"[{PH}] {name:20s} 예상 {sc['예상']:.3f} | 평일 {sc['평일']:.3f} | 토 {sc['토']:.3f} | "
          f"회차 행 MAE {mae(TR.tt_sec[t], p[t]):.1f} | 나머지 {mae(TR.tt_sec[~t], p[~t]):.3f}  ({time.time() - t0:.0f}초)", flush=True)
    return sc


def one(feats):
    return lambda f, v: lgb.LGBMRegressor(**LGB).fit(to_X(f, feats), f.tt_sec).predict(to_X(v, feats))


def small(f, v):
    ff, vv = f[f.is_turn2 == 1], v[v.is_turn2 == 1]
    p = pd.Series(np.nan, index=v.index)
    p[vv.index] = lgb.LGBMRegressor(**SMALL).fit(to_X(ff, SF), ff.tt_sec).predict(to_X(vv, SF))
    return p.values


t0 = time.time(); pA = oof(one(E5)); rA = report('A 기준 E30', pA, t0)
t0 = time.time(); pB = oof(one(E5 + H)); rB = report('B +회차 힌트', pB, t0)
t0 = time.time(); pS0 = oof(small); pS = pA.where(TR.is_turn2 == 0, pS0); rS = report('S 회차 전용 모델', pS, t0)
pM = pA.where(TR.is_turn2 == 0, (pA + pS0) / 2); rM = report('S2 A·S 반반(회차 행)', pM, time.time())
print(f"[{PH}] B - A = {rB['예상'] - rA['예상']:+.3f} | S - A = {rS['예상'] - rA['예상']:+.3f} | S2 - A = {rM['예상'] - rA['예상']:+.3f}")
