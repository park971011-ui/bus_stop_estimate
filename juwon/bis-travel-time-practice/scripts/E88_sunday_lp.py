"""E88 일요일 모델(E73) + 분 복원·회차 역산 (E87 피처)
  근거: E87 연습 일요일(참고) 29.979 → 분 복원 29.640, 둘 다 29.715
  LP는 데이터 조각(대회 train / 연습 train / 연습 test)마다 따로 풂 (차량 하루가 섞이지 않게)
  평소 시간: 학습 행은 학습 조각 전체에서 자기 날짜 빼고, 검증·예측 행은 학습 조각 전체로
  검증: E73과 같은 실습 노선 빼기 (LightGBM). 비교: E73 / +분 복원 / +둘 다
  채택 기준(미리 고정): E73보다 일요일 MAE 낮고 4개 노선 중 3개 이상 개선, 여럿이면 가장 낮은 것
실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E88_sunday_lp.py cv   (또는 make)
"""
import json, os, sys, time
import numpy as np, pandas as pd, lightgbm as lgb

MODE_ARG = sys.argv[1]
ROOT = '/Users/kimjuwon/Desktop/에모팀플/bis-travel-time-practice'
exec(open(f'{ROOT}/scripts/E80_sunday_cross.py').read().split('\nt0 = time.time()')[0])
MODE = MODE_ARG
sys.path.insert(0, f'{ROOT}/scripts')
from lp_dwell import lp_dwell
LPM, LPD = ['lp_min', 'lp_tod'], ['lp_D']


def smed(src_, apply):
    md = src_.groupby(SEGK).tt_sec.median()
    return apply[SEGK].join(md.rename('x'), on=SEGK).x.values


def lp_parts(tr_parts, ap_parts):
    pool = pd.concat(tr_parts)
    def one(d, med):
        r = lp_dwell(d, med, 0.3, 0.05); d = d.copy()
        d['lp_min'] = r.lp_min.values; d['lp_tod'] = d.arrive_hour + d.lp_min / 60; d['lp_D'] = r.lp_D.values
        return d
    out_tr = []
    for p in tr_parts:
        mf = np.full(len(p), np.nan)
        for dd in p.date.unique():
            m = (p.date == dd).values; mf[m] = smed(pool[pool.date != dd], p[m])
        out_tr.append(one(p, mf))
    return out_tr, [one(p, smed(pool, p)) for p in ap_parts]


SETS = {'E73': FB, '+분': FB + LPM, '+둘다': FB + LPM + LPD}
t0 = time.time()
if MODE == 'cv':
    L = []
    for r in sorted(PTE.route_nm.unique()):
        m = (PTE.route_nm == r).values
        P_all = pd.concat([PTR, PTE[~m]])
        trp, app = lp_parts([FTR, PTR, PTE[~m]], [PTE[m]])
        tr = with_px(trp, P_all)
        v = with_px([app[0][dow(app[0]) == 6]], P_all)
        row = {'row_id': v.row_id.values, 'route': r, 'y': v.tt_sec.values}
        for k, feats in SETS.items():
            row[k] = lgb.LGBMRegressor(**LGB).fit(Xf(tr, feats), tr.tt_sec).predict(Xf(v, feats))
        L.append(pd.DataFrame(row))
        print(f'[검증] 노선 {r}: ' + ' | '.join(f'{k} {np.abs(row["y"] - row[k]).mean():.3f}' for k in SETS) + f' ({time.time() - t0:.0f}초)', flush=True)
    L = pd.concat(L); L.to_parquet(f'{ROOT}/experiments/E88_loro.parquet')
    e = L.groupby('route').apply(lambda g: pd.Series({k: (g.y - g[k]).abs().mean() for k in SETS}))
    tot = {k: (L.y - L[k]).abs().mean() for k in SETS}
    ok = {k: tot[k] for k in SETS if k != 'E73' and tot[k] < tot['E73'] and (e[k] < e['E73']).sum() >= 3}
    best = min(ok, key=ok.get) if ok else None
    print('[검증] 일요일 전체: ' + ' | '.join(f'{k} {tot[k]:.3f}' for k in SETS) + ' | 좋아진 노선 ' + ' '.join(f'{k} {(e[k] < e["E73"]).sum()}/4' for k in SETS if k != 'E73'), flush=True)
    print(f'[검증] → {"채택: " + best if best else "채택 안 함"}', flush=True)
    json.dump({'best': best, 'tot': tot}, open(f'{ROOT}/experiments/E88_choice.json', 'w'), ensure_ascii=False)
