"""E43 E41(회차 힌트) 위에 세 가지를 대회 train 4-fold 검증으로 비교 (LightGBM 기준, 저장된 E41 검증 예측과 비교)
  1) log 모델 섞기: log(tt)를 예측하는 LightGBM(같은 피처·설정)을 E41 섞기 결과와 섞음 (log 비율 0.2 / 0.3 / 0.5)
  2) 토요일 가중치 2배: 학습할 때 토요일 행 가중치 2 (일요일용 아이디어. 검증엔 일요일이 없어서 평일 손해만 보임)
  3) 시간 섞기: 같은 모델로 arrive_hour를 h-1, h, h+1로 바꿔 예측 → 0.25 : 0.5 : 0.25
     (arrive_hour에서 바로 계산되는 is_peak, hour_from_first도 같이 바꿈. 운행·회차 힌트 피처는 그대로)
실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E43_log_satw_hourmix.py final
결과: 화면 점수 + experiments/oof_final_E43.parquet (약 20분, 제출 파일 없음)
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
TRAIN, TEST, test, to_X = G['TRAIN'], G['TEST'], G['test'], G['to_X']
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
F = G['feats_of'](['base', 'seg', 'time', 'trip']) + H
LGB = {**G['PARAMS'], 'min_child_samples': 10, 'learning_rate': 0.02, 'n_estimators': 1500}
DATES, DAY, cv_of, PEAK = G['DATES'], G['DAY'], G['cv_of'], G['PEAK']
old = pd.read_parquet(f'../experiments/oof_{PH}_E41.parquet').set_index('row_id').loc[TR.row_id.values]
y = TR.tt_sec.values
fmt = lambda r: ' | '.join(f'{k} {r[k]:.3f}' for k in ['월', '화', '수', '토', '평일', '예상'])
def rep(name, p):
    r = cv_of(pd.Series(np.asarray(p), index=TR.index))
    print(f'[{PH}] {name:30s} {fmt(r)}', flush=True); return r['예상']

def shift_hour(X, k):
    X = X.copy(); X['arrive_hour'] = X.arrive_hour + k
    X['is_peak'] = X.arrive_hour.isin(PEAK).astype(int)
    X['hour_from_first'] = X.arrive_hour * 60 - X.first_min
    return X

base, logm, satw, hmix = (pd.Series(np.nan, index=TR.index) for _ in range(4))
t0 = time.time()
for d in DATES:
    f, v = TR[TR.date != d], TR[TR.date == d]
    m = lgb.LGBMRegressor(**LGB).fit(to_X(f, F), f.tt_sec)
    base[v.index] = m.predict(to_X(v, F))
    hmix[v.index] = 0.25 * m.predict(to_X(shift_hour(v, -1), F)) + 0.5 * base[v.index].values + 0.25 * m.predict(to_X(shift_hour(v, 1), F))
    ml = lgb.LGBMRegressor(**LGB).fit(to_X(f, F), np.log1p(f.tt_sec))
    logm[v.index] = np.expm1(ml.predict(to_X(v, F)))
    w = np.where(pd.to_datetime(f.date).dt.day_of_week == 5, 2.0, 1.0)
    satw[v.index] = lgb.LGBMRegressor(**LGB).fit(to_X(f, F), f.tt_sec, sample_weight=w).predict(to_X(v, F))
    print(f'[{PH}] fold {DAY[pd.Timestamp(d).day_of_week]} 끝 ({time.time() - t0:.0f}초)', flush=True)

print(f'[{PH}] 검산: 다시 학습한 E41 LightGBM = 저장된 것? 최대 차이 {np.abs(base.values - old.lgb.values).max():.6f}', flush=True)
rep('E41 LightGBM (기준)', old.lgb.values)
rep('2) 토요일 가중치 2배', satw.values)
rep('3) 시간 섞기', hmix.values)
rep('1) log 모델 단독', logm.values)
rep('E41 섞기 0.8 (기준)', old.e41.values)
for a in [0.2, 0.3, 0.5]:
    rep(f'1) E41 섞기 + log {a}', (1 - a) * old.e41.values + a * logm.values)
rep('1+3) E41 섞기 + log 0.3, 시간섞기', 0.7 * (old.e41.values + (hmix.values - old.lgb.values) * 0.8) + 0.3 * logm.values)
pd.DataFrame({'row_id': TR.row_id.values, 'lgb_refit': base.values, 'log': logm.values, 'satw': satw.values, 'hourmix': hmix.values}).to_parquet(f'../experiments/oof_{PH}_E43.parquet')
print(f'[{PH}] 저장 ../experiments/oof_{PH}_E43.parquet ({time.time() - t0:.0f}초)')
