"""E84 일요일 모델(E73) + E83의 A(운행 길이 편향)·C(운행 규모) 피처
  근거: E83에서 기본 모델 기준 연습 일요일(참고) 29.979 → A 29.368, C 29.753, 전부 29.239. 평일엔 나빠졌지만 일요일은 처음 보는 요일이라 도움
  A: trip_span(운행이 시를 몇 번 넘겼나), trip_usual(운행 구간 평소 시간 합), trip_n, trip_lb = trip_span×3600/trip_usual
  C: svc_nveh(그날 노선 차량 수), svc_ntrip(차량 그날 운행 수), svc_nveh_h(같은 날·노선·시간 차량 수), 다른 날 평균 대비 비율 2개
     → 입력만 쓰므로 연습 전체·대회 전체(test 입력 포함)로 미리 계산
  평소 시간(구간 중앙값): 학습 행은 자기 날짜 뺀 학습 행으로, 검증·예측 행은 학습 행 전체로
  검증: E73과 같은 실습 노선 빼기 (LightGBM). 비교: E73 / +A / +C / +A+C
  채택 기준(미리 고정): E73보다 일요일 MAE 낮고 4개 노선 중 3개 이상 개선, 여럿이면 가장 낮은 것
실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E84_sunday_trip_svc.py cv   (또는 make)
"""
import json, os, sys, time
import numpy as np, pandas as pd, lightgbm as lgb

MODE_ARG = sys.argv[1]
ROOT = '/Users/kimjuwon/Desktop/에모팀플/bis-travel-time-practice'
exec(open(f'{ROOT}/scripts/E80_sunday_cross.py').read().split('\nt0 = time.time()')[0])
MODE = MODE_ARG
TK = ['date', 'vehicle_id', 'route_id', 'trip_no']
A = ['trip_span', 'trip_usual', 'trip_n', 'trip_lb']
C = ['svc_nveh', 'svc_ntrip', 'svc_nveh_h', 'svc_nveh_r', 'svc_nveh_h_r']


def add_svc(df):
    """C: 입력만 사용. df = 한 데이터(연습 전체 또는 대회 전체)"""
    s = df.copy()
    s['svc_nveh'] = s.groupby(['date', 'route_id']).vehicle_id.transform('nunique')
    s['svc_ntrip'] = s.groupby(['date', 'vehicle_id']).trip_no.transform('nunique')
    s['svc_nveh_h'] = s.groupby(['date', 'route_id', 'arrive_hour']).vehicle_id.transform('nunique')
    day = s.groupby(['date', 'route_id']).vehicle_id.nunique().groupby('route_id').mean()
    dh = s.groupby(['date', 'route_id', 'arrive_hour']).vehicle_id.nunique().groupby(['route_id', 'arrive_hour']).mean()
    s['svc_nveh_r'] = s.svc_nveh / s.route_id.map(day)
    s['svc_nveh_h_r'] = s.svc_nveh_h / pd.MultiIndex.from_arrays([s.route_id, s.arrive_hour]).map(dh).values
    return s


P2 = add_svc(pd.concat([PTR, PTE])); F2 = add_svc(pd.concat([FTR, FTE]))
PTR, PTE = P2.loc[P2.row_id.isin(PTR.row_id)], P2.loc[P2.row_id.isin(PTE.row_id)]
FTR, FTE = F2.loc[F2.row_id.isin(FTR.row_id)], F2.loc[F2.row_id.isin(FTE.row_id)]


def seg_med_map(src_, apply):
    md = src_.groupby(SEGK).tt_sec.median()
    return apply[SEGK].join(md.rename('x'), on=SEGK).x.values


def add_trip(df, med):
    d = df.copy(); d['_m'] = med; d['_m'] = d._m.fillna(np.nanmedian(med))
    g = d.groupby(TK)
    d['trip_span'] = g.arrive_hour.transform('max') - g.arrive_hour.transform('min')
    d['trip_usual'] = g._m.transform('sum'); d['trip_n'] = g._m.transform('size')
    d['trip_lb'] = d.trip_span * 3600 / d.trip_usual
    return d.drop(columns='_m')


def with_trip(tr, ap):
    mf = np.full(len(tr), np.nan)
    for dd in tr.date.unique():
        m = (tr.date == dd).values; mf[m] = seg_med_map(tr[~m], tr[m])
    return add_trip(tr, mf), add_trip(ap, seg_med_map(tr, ap))


SETS = {'E73': FB, '+A': FB + A, '+C': FB + C, '+A+C': FB + A + C}
t0 = time.time()
if MODE == 'cv':
    L = []
    for r in sorted(PTE.route_nm.unique()):
        m = (PTE.route_nm == r).values
        P_all = pd.concat([PTR, PTE[~m]])
        tr = with_px([FTR, PTR, PTE[~m]], P_all)
        v = with_px([PTE[m & (dow(PTE) == 6)]], P_all)
        tr, v = with_trip(tr, v)
        row = {'row_id': v.row_id.values, 'route': r, 'y': v.tt_sec.values}
        for k, feats in SETS.items():
            row[k] = lgb.LGBMRegressor(**LGB).fit(Xf(tr, feats), tr.tt_sec).predict(Xf(v, feats))
        L.append(pd.DataFrame(row))
        print(f'[검증] 노선 {r}: ' + ' | '.join(f'{k} {np.abs(row["y"] - row[k]).mean():.3f}' for k in SETS) + f' ({time.time() - t0:.0f}초)', flush=True)
    L = pd.concat(L); L.to_parquet(f'{ROOT}/experiments/E84_loro.parquet')
    e = L.groupby('route').apply(lambda g: pd.Series({k: (g.y - g[k]).abs().mean() for k in SETS}))
    tot = {k: (L.y - L[k]).abs().mean() for k in SETS}
    ok = {k: tot[k] for k in SETS if k != 'E73' and tot[k] < tot['E73'] and (e[k] < e['E73']).sum() >= 3}
    best = min(ok, key=ok.get) if ok else None
    print('[검증] 일요일 전체: ' + ' | '.join(f'{k} {tot[k]:.3f}' for k in SETS) + ' | 좋아진 노선 ' + ' '.join(f'{k} {(e[k] < e["E73"]).sum()}/4' for k in SETS if k != 'E73'), flush=True)
    print(f'[검증] → {"채택: " + best if best else "채택 안 함"}', flush=True)
    json.dump({'best': best, 'tot': tot}, open(f'{ROOT}/experiments/E84_choice.json', 'w'), ensure_ascii=False)
else:
    best = json.load(open(f'{ROOT}/experiments/E84_choice.json'))['best']; assert best, '검증 통과 못 함'
    FX2 = SETS[best]; print('조합', best, flush=True)
    P_all = pd.concat([PTR, PTE])
    ALL = with_px([FTR, PTR, PTE], P_all)
    sun = dow(FTE) == 6; FS = with_px([FTE[sun]], P_all)
    ALL, FS = with_trip(ALL, FS)
    diff = (FS.interval_sun != FS.interval_sat).values
    from catboost import CatBoostRegressor

    def fit(ws):
        w = np.where(dow(ALL) == 6, ws, 1.0)
        ml = lgb.LGBMRegressor(**LGB).fit(Xf(ALL, FX2), ALL.tt_sec, sample_weight=w)
        mc = CatBoostRegressor(**CB, cat_features=['route_id', 'seg_id']).fit(ALL[FX2], ALL.tt_sec, sample_weight=w)
        return lambda X: 0.8 * ml.predict(Xf(X, FX2)) + 0.2 * mc.predict(X[FX2])
    p = fit(1.0)(FS); p[diff] = fit(2.0)(FS[diff])
    base = pd.read_csv(f'{ROOT}/submissions/final_E79_cross_wd_E73sun.csv').set_index('row_id').tt_sec
    new = base.copy(); new.loc[FS.row_id.values] = np.round(p, 1)
    ss = pd.read_csv(f'{ROOT}/data/final/sample_submission.csv')
    sub = new.loc[ss.row_id].reset_index(); sub.columns = ['row_id', 'tt_sec']
    assert len(sub) == 111563 and sub.tt_sec.notna().all() and not sub.row_id.duplicated().any() and (sub.tt_sec >= 0).all()
    path = f'{ROOT}/submissions/final_E84_sunday_trip_svc.csv'; assert not os.path.exists(path); sub.to_csv(path, index=False)
    d = new.loc[FS.row_id.values].values - base.loc[FS.row_id.values].values
    print(f'저장 {path} | 목·금 = E79 | 일요일 E79 대비 평균 {d.mean():+.2f}초, |차이| {np.abs(d).mean():.2f}초 ({time.time() - t0:.0f}초)', flush=True)
