"""E80 = E73 일요일 모델 + 정각 넘김 피처(h_cross, h_cross_prev, E78). 연습 test 일요일(참고) 29.904 → 29.644
  검증: E73과 같은 실습 노선 빼기. 비교: E73(연습 기록) vs E73 + 정각 넘김. 기준: 일요일 MAE 개선 + 4개 노선 중 3개 이상

(아래는 E73 설명) E73 일요일 모델 + 같은 날 연습 기록 피처 (E69 A+B를 일요일 모델에)
  근거: E69에서 처음 보는 요일(토요일 fold)만 좋아짐 30.126 → 29.641. 일요일도 평소와 다른 날이라 같은 효과 기대.
  누출 방지: 각 행의 연습 기록 피처는 "자기 노선을 뺀" 연습 데이터로만 계산 (대회 노선 행은 연습 노선 전부 사용).
             검증 때 빠진 노선 r의 연습 test는 피처 계산에도 안 씀.
  규칙(미리 고정): 실습 노선 하나 빼기 검증(E60·E66과 같음), LightGBM만.
    기준 W0(E58 일요일 모델) 27.577보다 낮고 4개 노선 중 3개 이상 좋아지면 채택 → 그때 E67 방식(CatBoost 0.2, 배차 다른 노선 가중치)으로 제출 파일
실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E80_sunday_cross.py cv   (또는 make)
"""
import json, os, sys, time
import numpy as np, pandas as pd, lightgbm as lgb

MODE_ARG = sys.argv[1]
ROOT = '/Users/kimjuwon/Desktop/에모팀플/bis-travel-time-practice'
src = open(f'{ROOT}/scripts/E66_sunday_weight.py').read().split('\nt0 = time.time()')[0]
sys.argv = [sys.argv[0], 'cv']
exec(src)                                    # PTR, PTE, FTR, FTE, F0, LGB, CB, Xf, run_w, dow, CATS ...
MODE = MODE_ARG                              # 불러온 코드가 MODE를 덮어쓰므로 되돌림

SK = ['from_stop', 'to_stop']
PX = ['px_seg_hr', 'px_seg_r', 'px_seg_n', 'px_near_r', 'px_near_n', 'px_glob_r']
R_NEAR = 500.0


def seg_geo(df):
    g = df.groupby(SK)[['from_lat', 'from_lon', 'to_lat', 'to_lon']].first().reset_index()
    g['mlat'] = (g.from_lat + g.to_lat) / 2; g['mlon'] = (g.from_lon + g.to_lon) / 2
    g['vx'] = (g.to_lon - g.from_lon) * np.cos(np.radians(36.35)); g['vy'] = g.to_lat - g.from_lat
    n = np.hypot(g.vx, g.vy).replace(0, np.nan); g['vx'] /= n; g['vy'] /= n
    return g


def add_px(d, P):
    """d 행들에 연습 데이터 P(이미 자기 노선 뺀 것)로 같은 날 기록 피처를 붙임 (E69와 같은 계산)"""
    d = d.drop(columns=[c for c in PX if c in d.columns]).copy()
    P = P.copy()
    P['_usual'] = P.groupby(SK + ['arrive_hour']).tt_sec.transform('median')
    P['_lr'] = np.log(P.tt_sec / P._usual)
    a = P.groupby(SK + ['date', 'arrive_hour']).agg(px_seg_hr=('tt_sec', 'median'), px_seg_r=('_lr', 'mean'), px_seg_n=('tt_sec', 'size'))
    d = d.join(a, on=SK + ['date', 'arrive_hour'])
    PG, fg = seg_geo(P), seg_geo(d)
    out = []
    for r in fg.itertuples():
        dx = (PG.mlon.values - r.mlon) * 111320 * np.cos(np.radians(r.mlat)); dy = (PG.mlat.values - r.mlat) * 110540
        m = (np.hypot(dx, dy) <= R_NEAR) & (np.nan_to_num(PG.vx.values * r.vx + PG.vy.values * r.vy) > 0.7)
        out += [(r.from_stop, r.to_stop, PG.from_stop.iat[j], PG.to_stop.iat[j]) for j in np.where(m)[0]]
    pairs = pd.DataFrame(out, columns=['f_from', 'f_to', 'p_from', 'p_to'])
    ps = P.groupby(SK + ['date', 'arrive_hour'])._lr.mean().rename('lr').reset_index().rename(columns={'from_stop': 'p_from', 'to_stop': 'p_to'})
    nb = pairs.merge(ps, on=['p_from', 'p_to']).groupby(['f_from', 'f_to', 'date', 'arrive_hour']).lr.agg(px_near_r='mean', px_near_n='size')
    nb.index.names = SK + ['date', 'arrive_hour']
    d = d.join(nb, on=SK + ['date', 'arrive_hour']); d['px_near_n'] = d.px_near_n.fillna(0)
    d = d.join(P.groupby(['date', 'arrive_hour'])._lr.median().rename('px_glob_r'), on=['date', 'arrive_hour'])
    return d


def with_px(parts, P_all):
    """parts: 행 묶음 목록. 연습 행은 자기 노선을 뺀 P로, 대회 행은 P 전부로"""
    out = []
    for df in parts:
        if df.row_id.iloc[0].startswith('P'):
            out += [add_px(g, P_all[P_all.route_nm != rn]) for rn, g in df.groupby('route_nm')]
        else:
            out.append(add_px(df, P_all))
    return pd.concat(out)


XC = ['h_cross', 'h_cross_prev']


def add_cross(df):
    s = df.sort_values(['date', 'vehicle_id', 'route_id', 'trip_no', 'stop_seq'])
    g = s.groupby(['date', 'vehicle_id', 'route_id', 'trip_no'])
    ns, nh, ps, ph = g.stop_seq.shift(-1), g.arrive_hour.shift(-1), g.stop_seq.shift(1), g.arrive_hour.shift(1)
    s = s.copy()
    s['h_cross'] = np.where(ns == s.stop_seq + 1, nh - s.arrive_hour, np.nan)
    s['h_cross_prev'] = np.where(ps == s.stop_seq - 1, s.arrive_hour - ph, np.nan)
    return s.loc[df.index]


PTR, PTE, FTR, FTE = add_cross(PTR), add_cross(PTE), add_cross(FTR), add_cross(FTE)
FB = F0 + PX            # E73
FX = F0 + PX + XC       # E80
t0 = time.time()
if MODE == 'cv':
    rows, L = [], []
    for r in sorted(PTE.route_nm.unique()):
        m = (PTE.route_nm == r).values
        P_all = pd.concat([PTR, PTE[~m]])                  # 빠진 노선 r의 연습 test는 피처 계산에도 안 씀
        tr = with_px([FTR, PTR, PTE[~m]], P_all)
        v = with_px([PTE[m & (dow(PTE) == 6)]], P_all)
        p0 = lgb.LGBMRegressor(**LGB).fit(Xf(tr, FB), tr.tt_sec).predict(Xf(v, FB))
        p1 = lgb.LGBMRegressor(**LGB).fit(Xf(tr, FX), tr.tt_sec).predict(Xf(v, FX))
        e0, e1 = np.abs(v.tt_sec - p0).mean(), np.abs(v.tt_sec - p1).mean()
        print(f'[검증] 노선 {r}: E73 {e0:.3f} → +정각 넘김 {e1:.3f} ({e1 - e0:+.3f}) | 피처 있는 행: 같은 구간 {v.px_seg_hr.notna().mean():.0%}, 근처 {v.px_near_r.notna().mean():.0%} ({time.time() - t0:.0f}초)', flush=True)
        L.append(pd.DataFrame({'row_id': v.row_id.values, 'route': r, 'y': v.tt_sec.values, 'W0': p0, 'PX': p1}))
    L = pd.concat(L); L.to_parquet(f'{ROOT}/experiments/E80_loro.parquet')
    t0_, t1_ = (L.y - L.W0).abs().mean(), (L.y - L.PX).abs().mean()
    e = L.groupby('route').apply(lambda g: pd.Series({'W0': (g.y - g.W0).abs().mean(), 'PX': (g.y - g.PX).abs().mean()}))
    ok = t1_ < t0_ and (e.PX < e.W0).sum() >= 3
    print(f'[검증] 일요일 전체 E73 {t0_:.3f} → +정각 넘김 {t1_:.3f} ({t1_ - t0_:+.3f}) | 좋아진 노선 {(e.PX < e.W0).sum()}/4 → {"채택" if ok else "채택 안 함"}', flush=True)
    json.dump({'ok': bool(ok), 'w0': t0_, 'px': t1_}, open(f'{ROOT}/experiments/E80_choice.json', 'w'))
else:
    assert json.load(open(f'{ROOT}/experiments/E80_choice.json'))['ok'], '검증 통과 못 함'
    P_all = pd.concat([PTR, PTE])
    ALL = with_px([FTR, PTR, PTE], P_all)
    sun = dow(FTE) == 6; FS = with_px([FTE[sun]], P_all)
    diff = (FS.interval_sun != FS.interval_sat).values

    def fit(ws):
        w = np.where(dow(ALL) == 6, ws, 1.0)
        ml = lgb.LGBMRegressor(**LGB).fit(Xf(ALL, FX), ALL.tt_sec, sample_weight=w)
        from catboost import CatBoostRegressor
        mc = CatBoostRegressor(**CB, cat_features=['route_id', 'seg_id']).fit(ALL[FX], ALL.tt_sec, sample_weight=w)
        return lambda X: 0.8 * ml.predict(Xf(X, FX)) + 0.2 * mc.predict(X[FX])
    p = fit(1.0)(FS); p[diff] = fit(2.0)(FS[diff])            # E67과 같은 노선별 가중치
    e67 = pd.read_csv(f'{ROOT}/submissions/final_E79_cross_wd_E73sun.csv').set_index('row_id').tt_sec
    new = e67.copy(); new.loc[FS.row_id.values] = np.round(p, 1)
    ss = pd.read_csv(f'{ROOT}/data/final/sample_submission.csv')
    sub = new.loc[ss.row_id].reset_index(); sub.columns = ['row_id', 'tt_sec']
    assert len(sub) == 111563 and sub.tt_sec.notna().all() and not sub.row_id.duplicated().any() and (sub.tt_sec >= 0).all()
    path = f'{ROOT}/submissions/final_E80_sunday_cross.csv'; assert not os.path.exists(path); sub.to_csv(path, index=False)
    d = new.loc[FS.row_id.values].values - e67.loc[FS.row_id.values].values
    print(f'저장 {path} | 목·금 = E79 | 일요일 E73 대비 평균 {d.mean():+.2f}초, |차이| {np.abs(d).mean():.2f}초 ({time.time() - t0:.0f}초)', flush=True)
