"""E69 같은 날 연습 노선 기록 피처 (교수님 허용 확인 후 진행, 11/25)
  연습 데이터와 대회 데이터는 날짜가 같음 (train 9/23·24·25·28, test 9/26·27·29). 연습 test에는 정답 있음.
  → 대회 행마다 "그날 그 시간 같은/근처 도로가 실제로 얼마나 막혔나"를 연습 데이터에서 가져옴. 대회 정답은 안 씀.
  A) 같은 구간: 같은 날·같은 출발/도착 정류장·같은 시의 연습 통행시간 중앙값(px_seg_hr), 그 구간 평소(같은 시, 모든 날짜) 대비 비율(px_seg_r), 개수
  B) 근처 도로: 반경 R m 안, 진행 방향 비슷한(cos > 0.7) 연습 구간들의 그날·그 시 log(실제/평소) 평균(px_near_r), 개수(px_near_n)
     + 그날·그 시 연습 전체 log 비율 중앙값(px_glob_r)
  평소 = 연습 데이터 그 구간·그 시의 모든 날짜 중앙값
  검증: 대회 train 날짜별 4-fold, E41 LightGBM(25.227)과 비교. 연습 train도 같은 4일이라 test 때와 같은 조건.
실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E69_practice_sameday.py cv
"""
import json, os, re, sys, time
import numpy as np, pandas as pd, lightgbm as lgb

ROOT = '/Users/kimjuwon/Desktop/에모팀플/bis-travel-time-practice'
MODE = sys.argv[1] if len(sys.argv) > 1 else 'cv'
R_NEAR = 500.0
src = open(f'{ROOT}/scripts/E65_gap_minute.py').read()
src = src.split('\ndef add_gap')[0]
sys.argv = [sys.argv[0], 'final']
exec(src)                                   # TRAIN, TEST, TR(E41 피처), F, LGB, DATES, cv_of, to_X

P = pd.concat([pd.read_parquet(f'{ROOT}/data/practice/train.parquet'), pd.read_parquet(f'{ROOT}/data/practice/test.parquet')])
SK = ['from_stop', 'to_stop']
P['_usual'] = P.groupby(SK + ['arrive_hour']).tt_sec.transform('median')
P['_lr'] = np.log(P.tt_sec / P._usual)


def seg_geo(df):
    g = df.groupby(SK)[['from_lat', 'from_lon', 'to_lat', 'to_lon']].first().reset_index()
    g['mlat'] = (g.from_lat + g.to_lat) / 2; g['mlon'] = (g.from_lon + g.to_lon) / 2
    g['vx'] = (g.to_lon - g.from_lon) * np.cos(np.radians(36.35)); g['vy'] = g.to_lat - g.from_lat
    n = np.hypot(g.vx, g.vy).replace(0, np.nan); g['vx'] /= n; g['vy'] /= n
    return g


PG = seg_geo(P)


def near_pairs(fg):
    """대회 구간 × 연습 구간 쌍 (거리 R 이하, 방향 비슷)"""
    out = []
    for r in fg.itertuples():
        dx = (PG.mlon.values - r.mlon) * 111320 * np.cos(np.radians(r.mlat)); dy = (PG.mlat.values - r.mlat) * 110540
        dist = np.hypot(dx, dy); cos = PG.vx.values * r.vx + PG.vy.values * r.vy
        m = (dist <= R_NEAR) & (np.nan_to_num(cos) > 0.7)
        for j in np.where(m)[0]:
            out.append((r.from_stop, r.to_stop, PG.from_stop.iat[j], PG.to_stop.iat[j]))
    return pd.DataFrame(out, columns=['f_from', 'f_to', 'p_from', 'p_to'])


def add_px(df):
    d = df.copy()
    # A) 같은 구간
    a = P.groupby(SK + ['date', 'arrive_hour']).agg(px_seg_hr=('tt_sec', 'median'), px_seg_r=('_lr', 'mean'), px_seg_n=('tt_sec', 'size'))
    d = d.join(a, on=SK + ['date', 'arrive_hour'])
    # B) 근처 도로
    pairs = near_pairs(seg_geo(d))
    ps = P.groupby(SK + ['date', 'arrive_hour'])._lr.mean().rename('lr').reset_index().rename(columns={'from_stop': 'p_from', 'to_stop': 'p_to'})
    j = pairs.merge(ps, on=['p_from', 'p_to'])
    nb = j.groupby(['f_from', 'f_to', 'date', 'arrive_hour']).lr.agg(px_near_r='mean', px_near_n='size')
    nb.index.names = SK + ['date', 'arrive_hour']
    d = d.join(nb, on=SK + ['date', 'arrive_hour'])
    d['px_near_n'] = d.px_near_n.fillna(0)
    g = P.groupby(['date', 'arrive_hour'])._lr.median().rename('px_glob_r')
    d = d.join(g, on=['date', 'arrive_hour'])
    return d


t0 = time.time()
TRX = add_px(TR)
print(f'피처 만들기 {time.time() - t0:.0f}초 | 같은 구간 값 있는 행 {TRX.px_seg_hr.notna().mean():.1%} | 근처 도로 값 있는 행 {TRX.px_near_r.notna().mean():.1%}', flush=True)
A = ['px_seg_hr', 'px_seg_r', 'px_seg_n']; B = ['px_near_r', 'px_near_n', 'px_glob_r']
old = pd.read_parquet(f'{ROOT}/experiments/oof_final_E41.parquet').set_index('row_id').loc[TRX.row_id.values]
y = TRX.tt_sec.values
hasA = TRX.px_seg_hr.notna().values; hasB = TRX.px_near_r.notna().values
res = {'E41 LightGBM': old.lgb.values}
for name, extra in [('A 같은 구간', A), ('B 근처 도로', B), ('C 둘 다', A + B)]:
    o = pd.Series(np.nan, index=TRX.index)
    for dd in DATES:
        fm = (TRX.date != dd).values; f, v = TRX[fm], TRX[~fm]
        o[v.index] = lgb.LGBMRegressor(**LGB).fit(to_X(f, F + extra), f.tt_sec).predict(to_X(v, F + extra))
    res[name] = o.values
    print(f'{name} 끝 ({time.time() - t0:.0f}초)', flush=True)
rows = []
for n, p in res.items():
    r = cv_of(pd.Series(p, index=TRX.index)); e = np.abs(y - p)
    rows.append({'실험': n, '월': r['월'], '화': r['화'], '수': r['수'], '토': r['토'], '평일': (r['월'] + r['화'] + r['수']) / 3, '예상': r['예상'],
                 '같은구간 행': e[hasA].mean(), '근처도로 행': e[hasB].mean(), '근처 없는 행': e[~hasB].mean()})
T = pd.DataFrame(rows).set_index('실험').round(3)
print(T.to_string(), flush=True)
pd.DataFrame({'row_id': TRX.row_id.values, **{k: v for k, v in res.items()}}).to_parquet(f'{ROOT}/experiments/E69_oof.parquet')
T.to_csv(f'{ROOT}/experiments/E69_cv_table.csv')
