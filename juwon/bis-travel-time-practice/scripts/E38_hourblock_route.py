"""E38 두 가지 검증 비교 (LightGBM E30 설정만, CatBoost 없이 빠르게):
  A) 기준: E30 (E5 피처 25개)
  B) +시간 칸 피처: 한 운행 안에서 같은 arrive_hour에 들어간 구간 수·거리 합 등
  C) 노선별 모델: E5 피처, 노선마다 LightGBM 따로

실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E38_hourblock_route.py final   (또는 practice)
결과: 화면에 검증 점수만 출력 (제출 파일 안 만듦). 약 15~20분
주의: 노트북 cv_feature_experiments.ipynb 의 셀(2,4,6,8,10,27,29,41번)을 실행해서 피처·모델 함수를 가져옴.
"""
import json, os, sys, time
import numpy as np, pandas as pd, lightgbm as lgb
os.chdir('/Users/kimjuwon/Desktop/에모팀플/bis-travel-time-practice/notebooks')
PH = sys.argv[1]
nb = json.load(open('cv_feature_experiments.ipynb')); G = {}
for i in [2, 4, 6, 8, 10, 27, 29, 41]:
    src = ''.join(nb['cells'][i]['source'])
    if i == 2: src = __import__('re').sub(r"PHASE = '\w+'", f"PHASE = '{PH}'", src, count=1)
    exec(src, G)
TRAIN, TEST = G['TRAIN'], G['TEST']


def add_hourblock(df):
    """운행(trip) 안에서 arrive_hour가 같은 구간 묶음 = 시간 칸. 입력(arrive_hour)만으로 계산."""
    df = df.copy()
    trip = ['date', 'vehicle_id', 'route_id', 'direction', 'trip_no']
    blk = trip + ['arrive_hour']
    g = df.groupby(blk)
    df['hb_n'] = g.stop_seq.transform('size')                     # 이 시간 칸에 든 구간 수
    df['hb_dist'] = g.dist_m.transform('sum')                     # 이 시간 칸 거리 합
    df['hb_pos'] = df.stop_seq - g.stop_seq.transform('min')      # 칸 안에서 몇 번째
    df['hb_left'] = g.stop_seq.transform('max') - df.stop_seq     # 칸 안에서 남은 구간
    t = df.groupby(trip).arrive_hour
    inner = (df.arrive_hour > t.transform('min')) & (df.arrive_hour < t.transform('max'))   # 앞뒤 칸이 다 있는 칸 = 꽉 찬 1시간
    df['hb_full'] = inner.astype(int)
    df['hb_est'] = np.where(inner, 3600 * df.dist_m / df.hb_dist, np.nan)                  # 1시간을 거리 비율로 나눈 값
    df['hb_per_seg'] = np.where(inner, 3600 / df.hb_n, np.nan)
    return df


HB = ['hb_n', 'hb_dist', 'hb_pos', 'hb_left', 'hb_full', 'hb_est', 'hb_per_seg']
TR = add_hourblock(TRAIN)
G['TRAIN'] = TR
LGB = {**G['PARAMS'], 'min_child_samples': 10, 'learning_rate': 0.02, 'n_estimators': 1500}
E5 = G['feats_of'](['base', 'seg', 'time', 'trip'])
DATES, DAY, summarize, mae, to_X = G['DATES'], G['DAY'], G['summarize'], G['mae'], G['to_X']
print(f'[{PH}] 꽉 찬 시간 칸 비율: train {TR.hb_full.mean():.1%} | test {add_hourblock(TEST).hb_full.mean():.1%}', flush=True)


def cv(fit_pred, name):
    t0 = time.time(); sc = {}
    for d in DATES:
        f, v = TR[TR.date != d], TR[TR.date == d]
        sc[DAY[pd.Timestamp(d).day_of_week]] = mae(v.tt_sec, fit_pred(f, v))
    r = summarize(sc)
    print(f"[{PH}] {name:22s} 예상 {r['예상']:.3f} | 평일 {r['평일']:.3f} | 토 {r['토']:.3f} | "
          + ' '.join(f'{k} {sc[k]:.3f}' for k in sc) + f'  ({time.time() - t0:.0f}초)', flush=True)
    return r


def one(feats):
    return lambda f, v: lgb.LGBMRegressor(**LGB).fit(to_X(f, feats), f.tt_sec).predict(to_X(v, feats))


def per_route(f, v):
    p = pd.Series(np.nan, index=v.index)
    for r, vv in v.groupby('route_id'):
        ff = f[f.route_id == r]
        p[vv.index] = lgb.LGBMRegressor(**LGB).fit(to_X(ff, E5), ff.tt_sec).predict(to_X(vv, E5))
    return p.values


out = {}
out['A'] = cv(one(E5), 'A 기준 E30')
out['B'] = cv(one(E5 + HB), 'B +시간 칸')
out['C'] = cv(per_route, 'C 노선별 모델')
print(f"[{PH}] B - A = {out['B']['예상'] - out['A']['예상']:+.3f} | C - A = {out['C']['예상'] - out['A']['예상']:+.3f}")
