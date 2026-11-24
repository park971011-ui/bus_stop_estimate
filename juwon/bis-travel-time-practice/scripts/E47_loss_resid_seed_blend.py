"""E47 E41 위에서 한 번에 하나씩 바꾸기 (E41 피처 그대로, 블렌드 비율은 미리 고정)
  0) E41: LightGBM(regression_l1 = 이미 L1) 0.8 + CatBoost(E31) 0.2, seed 0, 피처 E5 25개 + 회차 힌트 10개, 타깃 tt_sec
  1) 손실: Huber(alpha 25 = 타깃 MAE 근처) LightGBM  (L1은 E41 LightGBM 그대로)
  2) 잔차 학습: 기준값 = 학습 날짜의 노선×시간대 중앙값(행 30개 미만이면 노선 중앙값), 타깃 = 실제 − 기준값, 1번에서 좋았던 손실
  3) seed 5개 평균: 1·2번 중 최고 설정
  4) CatBoost(MAE, depth 6, 범주형 = 노선·구간·주말 여부·시간대) 새로 학습 → 3번 LightGBM 0.5 + CatBoost 0.5
     (MLP는 대회 규칙 "앙상블 계열 모델만"에 어긋나서 뺌. 요일 범주는 test 요일이 학습에 없어서 is_weekend로 대신함)
  5) 사후 보정: 지금까지 최고 모델의 노선×시간대 검증 잔차 중앙값 × 0.5를 더함 (검증 날짜를 빼고 계산, 행 30개 미만 칸은 0)
선택은 대회 train 날짜별 검증(E41과 같은 분할) '예상' 점수로만. 연습 데이터는 마지막 확인 한 번.
실행: cd bis-travel-time-practice
  /opt/miniconda3/envs/bis/bin/python scripts/E47_loss_resid_seed_blend.py final cv      → 표 + experiments/E47_choice.json, oof_final_E47.parquet (약 40분)
  /opt/miniconda3/envs/bis/bin/python scripts/E47_loss_resid_seed_blend.py practice make → 고른 조합을 연습 데이터로 확인 (+ practice 파일)
  /opt/miniconda3/envs/bis/bin/python scripts/E47_loss_resid_seed_blend.py final make    → submissions/final_E47_best.csv
주의: 노트북 cv_feature_experiments.ipynb 의 셀(2,4,6,8,10,27,29,41번)을 실행해서 피처·모델 함수를 가져옴. E41 코드·결과는 안 건드림.
"""
import json, os, re, sys, time
import numpy as np, pandas as pd, lightgbm as lgb
from catboost import CatBoostRegressor
os.chdir('/Users/kimjuwon/Desktop/에모팀플/bis-travel-time-practice/notebooks')
PH, MODE = sys.argv[1], sys.argv[2]
nb = json.load(open('cv_feature_experiments.ipynb')); G = {}
for i in [2, 4, 6, 8, 10, 27, 29, 41]:
    src = ''.join(nb['cells'][i]['source'])
    if i == 2:
        src = re.sub(r"PHASE = '\w+'", f"PHASE = '{PH}'", src, count=1); assert f"PHASE = '{PH}'" in src
    exec(src, G)
TRAIN, TEST, test, to_X = G['TRAIN'], G['TEST'], G['test'], G['to_X']
DATES, DAY, cv_of = G['DATES'], G['DAY'], G['cv_of']
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

TR, TE = add_turn_hint(TRAIN), add_turn_hint(TEST)
F = G['feats_of'](['base', 'seg', 'time', 'trip']) + H
LGB = {**G['PARAMS'], 'min_child_samples': 10, 'learning_rate': 0.02, 'n_estimators': 1500}
CB_CATS = ['route_id', 'seg_id', 'is_weekend', 'arrive_hour']
CBN = dict(loss_function='MAE', iterations=2000, learning_rate=0.08, depth=6, random_seed=0, verbose=0, thread_count=-1)
OBJ = {'l1': {}, 'huber': {'objective': 'huber', 'alpha': 25.0}}
MIN_N, HUB = 30, None


def base_of(fit, apply):
    """노선×시간대 중앙값 (학습 행만), 행이 MIN_N개 미만인 칸은 노선 중앙값"""
    g = fit.groupby(['route_id', 'arrive_hour']).tt_sec.agg(['median', 'size'])
    rh = g['median'].where(g['size'] >= MIN_N)
    r = fit.groupby('route_id').tt_sec.median()
    v = apply[['route_id', 'arrive_hour']].join(rh.rename('b'), on=['route_id', 'arrive_hour']).b
    return v.fillna(apply.route_id.map(r)).fillna(fit.tt_sec.median()).values


def lgb_pred(fit, applies, obj, resid, seed):
    p = {**LGB, **OBJ[obj], 'random_state': seed}
    if resid:
        m = lgb.LGBMRegressor(**p).fit(to_X(fit, F), fit.tt_sec - base_of(fit, fit))
        return [m.predict(to_X(a, F)) + base_of(fit, a) for a in applies]
    m = lgb.LGBMRegressor(**p).fit(to_X(fit, F), fit.tt_sec)
    return [m.predict(to_X(a, F)) for a in applies]


def cb_pred(fit, applies):
    m = CatBoostRegressor(**CBN, cat_features=CB_CATS).fit(fit[F], fit.tt_sec)
    return [m.predict(a[F]) for a in applies]


def model_pred(fit, applies, cfg):
    """cfg: obj, resid, seeds, cb(블렌드 여부)"""
    L = [lgb_pred(fit, applies, cfg['obj'], cfg['resid'], s) for s in range(cfg['seeds'])]
    out = [np.mean([l[i] for l in L], axis=0) for i in range(len(applies))]
    if cfg['cb']:
        C = cb_pred(fit, applies)
        out = [0.5 * o + 0.5 * c for o, c in zip(out, C)]
    return out


def oof_of(cfg, tag):
    o = pd.Series(np.nan, index=TR.index); t0 = time.time()
    for d in DATES:
        m = TR.date == d
        o[m] = model_pred(TR[~m], [TR[m]], cfg)[0]
    print(f'    ({tag} {time.time() - t0:.0f}초)', flush=True)
    return o.values


def corr_table(df, res):
    g = pd.DataFrame({'route_id': df.route_id.values, 'arrive_hour': df.arrive_hour.values, 'r': res}).groupby(['route_id', 'arrive_hour']).r.agg(['median', 'size'])
    return g['median'].where(g['size'] >= MIN_N).fillna(0.0)


def corr_apply(df, tab):
    return df[['route_id', 'arrive_hour']].join(tab.rename('c'), on=['route_id', 'arrive_hour']).c.fillna(0.0).values


def lodo_corrected(oof):
    """검증 날짜 d의 보정값은 d를 뺀 나머지 날짜 검증 잔차로만 계산"""
    out = oof.copy(); res = TR.tt_sec.values - oof
    for d in DATES:
        m = (TR.date == d).values
        out[m] = oof[m] + 0.5 * corr_apply(TR[m], corr_table(TR[~m], res[~m]))
    return out


DESC = lambda c: (f"LightGBM {'Huber' if c['obj'] == 'huber' else 'L1'}{' 잔차' if c['resid'] else ''} seed {c['seeds']}"
                  + (' + CatBoost 0.5' if c['cb'] else '') + (' + 사후 보정 0.5' if c.get('corr') else ''))

if MODE == 'cv':
    old = pd.read_parquet('../experiments/oof_final_E41.parquet').set_index('row_id').loc[TR.row_id.values]
    REF = cv_of(pd.Series(old.e41.values, index=TR.index))['예상']
    rows, P = [], {}
    def rec(name, p, cfg=None):
        r = cv_of(pd.Series(p, index=TR.index))
        rows.append({'실험': name, **{k: round(r[k], 3) for k in ['월', '화', '수', '토']}, '4일 평균': round(np.mean([r[k] for k in ['월', '화', '수', '토']]), 3),
                     '평일': round(r['평일'], 3), '예상': round(r['예상'], 3), 'E41 최종 대비': round(r['예상'] - REF, 3)})
        P[name] = (p, cfg, r['예상'])
        print(f"[{PH}] {name:42s} 월 {r['월']:.3f} 화 {r['화']:.3f} 수 {r['수']:.3f} 토 {r['토']:.3f} | 예상 {r['예상']:.3f} ({r['예상'] - REF:+.3f})", flush=True)
    rec('E41 최종 (LGB 0.8 + CB 0.2)', old.e41.values)
    rec('1a E41 LightGBM = L1', old.lgb.values, {'obj': 'l1', 'resid': False, 'seeds': 1, 'cb': False})
    c = {'obj': 'huber', 'resid': False, 'seeds': 1, 'cb': False}; rec('1b LightGBM Huber 25', oof_of(c, '1b'), c)
    best1 = min(['1a E41 LightGBM = L1', '1b LightGBM Huber 25'], key=lambda k: P[k][2]); obj = P[best1][1]['obj']
    print(f'[{PH}] → 1번 최고 손실: {obj}', flush=True)
    c = {'obj': obj, 'resid': True, 'seeds': 1, 'cb': False}; rec(f'2 잔차 학습 ({obj})', oof_of(c, '2'), c)
    best12 = min([best1, f'2 잔차 학습 ({obj})'], key=lambda k: P[k][2]); c = {**P[best12][1], 'seeds': 5}
    print(f'[{PH}] → 1·2번 최고: {best12} → seed 5개', flush=True)
    rec(f'3 seed 5개 ({best12[:2].strip()})', oof_of(c, '3'), c); n3 = rows[-1]['실험']
    c4 = {**P[n3][1], 'cb': True}
    # CatBoost만 따로 학습 → 3번 LightGBM 검증 예측과 0.5 : 0.5
    o_cb = pd.Series(np.nan, index=TR.index); t0 = time.time()
    for d in DATES:
        m = TR.date == d; o_cb[m] = cb_pred(TR[~m], [TR[m]])[0]
    print(f'    (4 CatBoost {time.time() - t0:.0f}초)', flush=True)
    rec('4a 새 CatBoost 단독 (depth 6)', o_cb.values)
    rec('4b 3번 LGB 0.5 + 새 CatBoost 0.5', 0.5 * P[n3][0] + 0.5 * o_cb.values, c4)
    bestm = min([k for k in P if P[k][1] is not None], key=lambda k: P[k][2])
    print(f'[{PH}] → 보정할 최고 모델: {bestm}', flush=True)
    rec(f'5 사후 보정 ({bestm[:2].strip()})', lodo_corrected(P[bestm][0]), {**P[bestm][1], 'corr': True})
    tab = pd.DataFrame(rows).set_index('실험'); print(tab.to_string(), flush=True)
    tab.to_csv('../experiments/E47_cv_table.csv')
    win = min([k for k in P if P[k][1] is not None], key=lambda k: P[k][2])
    choice = {'name': win, 'cfg': P[win][1], 'desc': DESC(P[win][1]), '예상': P[win][2], 'E41 최종': REF}
    json.dump(choice, open('../experiments/E47_choice.json', 'w'), ensure_ascii=False, indent=1)
    pd.DataFrame({'row_id': TR.row_id.values, **{k: v[0] for k, v in P.items()}}).to_parquet('../experiments/oof_final_E47.parquet')
    print(f"[{PH}] 최고 조합: {win} = {choice['desc']} | 예상 {choice['예상']:.3f} (E41 최종 {REF:.3f}, {choice['예상'] - REF:+.3f})", flush=True)
else:
    ch = json.load(open('../experiments/E47_choice.json')); cfg = ch['cfg']
    print(f"[{PH}] 고른 조합 (대회 검증으로 고름): {ch['desc']}", flush=True)
    sun = pd.to_datetime(TE.date).dt.day_of_week.values == 6
    S = TE.copy(); S['is_weekend'] = 1; S['interval'] = S.interval_sat
    W = TE.copy(); W['is_weekend'] = 0; W['interval'] = W.interval_wd
    t0 = time.time()
    pB, pS, pW = model_pred(TR, [TE, S, W], cfg)
    p = np.where(sun, 0.5 * pS + 0.5 * pW, pB)                        # 일요일 반반 (E41과 같음)
    if cfg.get('corr'):
        oof = oof_of(cfg, '보정용 검증 예측')
        p = p + 0.5 * corr_apply(TE, corr_table(TR, TR.tt_sec.values - oof))
    print(f'[{PH}] 학습·예측 {time.time() - t0:.0f}초', flush=True)
    sub = pd.DataFrame({'row_id': TE.row_id.values, 'tt_sec': np.round(p, 1)}).set_index('row_id').loc[test.row_id].reset_index()
    assert sub.tt_sec.notna().all() and not sub.row_id.duplicated().any()
    if os.path.exists(f'../data/{PH}/sample_submission.csv'):
        ss = pd.read_csv(f'../data/{PH}/sample_submission.csv'); assert (sub.row_id.values == ss.row_id.values).all()
    print(f'[{PH}] 음수 예측 {(sub.tt_sec < 0).sum()}개', flush=True)
    assert PH != 'final' or (sub.tt_sec >= 0).all()
    path = f'../submissions/{PH}_E47_best.csv'; sub.to_csv(path, index=False); print(f'[{PH}] 저장 {path} {sub.shape}', flush=True)
    if 'tt_sec' in test.columns:
        old = pd.read_csv(f'../submissions/{PH}_E41_turnhint_sunmix.csv').set_index('row_id').tt_sec.loc[test.row_id].values
        y = test.tt_sec.values; dw = pd.to_datetime(test.date).dt.day_of_week.values
        for n, q in [('E41', old), ('E47', sub.tt_sec.values)]:
            a = np.abs(y - q)
            print(f'[{PH}] {n} 연습 test {a.mean():.3f} | 목 {a[dw == 3].mean():.3f} | 금 {a[dw == 4].mean():.3f} | 일 {a[dw == 6].mean():.3f}', flush=True)
