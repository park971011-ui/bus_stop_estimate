"""E66 일요일 모델 가중치 실험 (E60 일요일 모델 = 대회 train + 실습 train + 실습 test, LightGBM 0.8 + CatBoost 0.2)
  가설: 학습 행 대부분이 평일이라 일요일(실습 test 일요일)·토요일 행이 모델에 덜 반영됨 → 가중치를 키우면 일요일 예측이 좋아질 것
  규칙(미리 고정):
    1단계 (LightGBM만, 빠르게): 가중치 없음(W0) vs 일요일 ×2, ×3, ×5 vs 토·일 ×2, ×3
    채택 기준: W0보다 일요일 MAE가 낮고, 실습 노선 4개 중 3개 이상 개선된 것 중 가장 낮은 것
    2단계: 채택된 가중치로 E60 조합(LightGBM 0.8 + CatBoost 0.2, CatBoost에도 같은 가중치) 한 번 확인 → E60 27.511과 비교
  검증: 실습 노선 하나 r을 빼고(r의 실습 test 전부 제외, r의 실습 train은 포함) 학습 → r의 실습 test 일요일 예측, 4번 (E60과 같음)
실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E66_sunday_weight.py cv   (또는 make)
결과: cv → 화면 + experiments/E66_loro.parquet, E66_choice.json / make → submissions/final_E66_sunday_weight.csv
"""
import json, os, sys, time
import numpy as np, pandas as pd, lightgbm as lgb

MODE = sys.argv[1]
ROOT = '/Users/kimjuwon/Desktop/에모팀플/bis-travel-time-practice'
# E60의 데이터 준비·피처·모델 설정 부분을 그대로 가져옴 (t0 이전까지)
src = open(f'{ROOT}/scripts/E60_sunday_refine.py').read().split('\nt0 = time.time()')[0]
sys.argv = [sys.argv[0], 'cv']
exec(src)
from catboost import CatBoostRegressor

VARIANTS = {  # 이름: (일요일 가중치, 토요일 가중치)
    'W0 가중치 없음': (1, 1),
    'W1 일요일 x2': (2, 1),
    'W2 일요일 x3': (3, 1),
    'W3 일요일 x5': (5, 1),
    'W4 토·일 x2': (2, 2),
    'W5 토·일 x3': (3, 3),
}


def wts(df, ws, wsat):
    d = dow(df)
    return np.where(d == 6, ws, np.where(d == 5, wsat, 1)).astype(float)


def run_w(tr, ap, ws, wsat, cb=False):
    w = wts(tr, ws, wsat)
    p = lgb.LGBMRegressor(**LGB).fit(Xf(tr, F0), tr.tt_sec, sample_weight=w).predict(Xf(ap, F0))
    if cb:
        m = CatBoostRegressor(**CB, cat_features=['route_id', 'seg_id'])
        p = 0.8 * p + 0.2 * m.fit(tr[F0], tr.tt_sec, sample_weight=w).predict(ap[F0])
    return p


t0 = time.time()
if MODE == 'cv':
    sets = {}
    for r in sorted(PTE.route_nm.unique()):
        m = (PTE.route_nm == r).values
        sets[r] = (pd.concat([FTR, PTR, PTE[~m]]), PTE[m & (dow(PTE) == 6)])
    L = pd.concat([pd.DataFrame({'row_id': v.row_id.values, 'route': r, 'y': v.tt_sec.values}) for r, (_, v) in sets.items()])

    def ev(name, fn):
        L[name] = np.concatenate([fn(*sets[r]) for r in sets])
        e = L.groupby('route').apply(lambda g: (g.y - g[name]).abs().mean())
        tot = (L.y - L[name]).abs().mean()
        print(f'[검증] {name:24s} 일요일 {tot:.3f} | ' + ' '.join(f'{k} {x:.3f}' for k, x in e.items()) + f'  ({time.time() - t0:.0f}초)', flush=True)
        return tot, e

    res = {k: ev(k, lambda tr, v, ws=ws, wsat=wsat: run_w(tr, v, ws, wsat)) for k, (ws, wsat) in VARIANTS.items()}
    b_tot, b_e = res['W0 가중치 없음']
    ok = {k: t for k, (t, e) in res.items() if k != 'W0 가중치 없음' and t < b_tot and (e < b_e).sum() >= 3}
    if not ok:
        print('[검증] → 기준 통과한 가중치 없음. 채택 안 함', flush=True); cfg = None
    else:
        best = min(ok, key=ok.get); ws, wsat = VARIANTS[best]
        print(f'[검증] → 1단계 채택: {best} ({b_tot:.3f} → {ok[best]:.3f})', flush=True)
        t2, _ = ev(f'S2 {best} + CatBoost 0.2', lambda tr, v: run_w(tr, v, ws, wsat, cb=True))
        e60 = pd.read_parquet(f'{ROOT}/experiments/E60_loro.parquet').set_index('row_id')['S2a + CatBoost 0.2']
        t60 = (L.y.values - e60.loc[L.row_id].values).__abs__().mean()
        print(f'[검증] E60 (가중치 없음 + CatBoost 0.2) {t60:.3f} → E66 {t2:.3f} ({t2 - t60:+.3f})', flush=True)
        cfg = {'name': best, 'ws': ws, 'wsat': wsat, 'cb': True, 'cv': t2, 'e60': t60}
    L.to_parquet(f'{ROOT}/experiments/E66_loro.parquet')
    json.dump(cfg, open(f'{ROOT}/experiments/E66_choice.json', 'w'), ensure_ascii=False)
else:
    cfg = json.load(open(f'{ROOT}/experiments/E66_choice.json')); assert cfg, '채택된 가중치 없음'; print('조합', cfg, flush=True)
    ALL = pd.concat([FTR, PTR, PTE]); sun = dow(FTE) == 6
    p = run_w(ALL, FTE[sun], cfg['ws'], cfg['wsat'], cb=cfg['cb'])
    e60 = pd.read_csv(f'{ROOT}/submissions/final_E60_sunday_refine.csv').set_index('row_id').tt_sec
    new = e60.copy(); ids = FTE.row_id.values[sun]; new.loc[ids] = np.round(p, 1)
    ss = pd.read_csv(f'{ROOT}/data/final/sample_submission.csv')
    sub = new.loc[ss.row_id].reset_index(); sub.columns = ['row_id', 'tt_sec']
    assert len(sub) == 111563 and (sub.row_id.values == ss.row_id.values).all()
    assert sub.tt_sec.notna().all() and not sub.row_id.duplicated().any() and (sub.tt_sec >= 0).all()
    path = f'{ROOT}/submissions/final_E66_sunday_weight.csv'; assert not os.path.exists(path); sub.to_csv(path, index=False)
    d = new.loc[ids].values - e60.loc[ids].values
    print(f'저장 {path} | 목·금 = E60(=E58) | 일요일 E60 대비 평균 {d.mean():+.2f}초, |차이| {np.abs(d).mean():.2f}초 ({time.time() - t0:.0f}초)', flush=True)
