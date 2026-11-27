"""E77 RandomForest·ExtraTrees 섞기 (대회 train 날짜별 4-fold)
  피처: E41 35개 (범주형은 숫자 코드) + 구간 평소 시간 te_seg, te_seg_hour, te_seg_speed (fold마다 학습 날짜로, 학습 행은 자기 날짜 빼고)
  타깃: log(tt_sec)로 학습 → exp로 되돌림 (MSE 학습이라 평균 대신 중앙값에 가까운 값을 예측하도록)
  결측: -1로 채움
  비교: E41 섞기(LGB 0.8 + CB 0.2, OOF 25.229) × (1-w) + 새 모델 × w, w = 0.1, 0.2, 0.3
  채택 기준(미리 고정): 예상 점수가 좋아지고 평일 3일 중 2일 이상 좋아짐
실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E77_rf_et.py
"""
import sys, time, json
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

ROOT = '/Users/kimjuwon/Desktop/에모팀플/bis-travel-time-practice'
src = open(f'{ROOT}/scripts/E65_gap_minute.py').read().split('\ndef add_gap')[0]
sys.argv = [sys.argv[0], 'final']
exec(src)                                    # TR(E41 피처), F, DATES, cv_of, G

TE = G['FEATS']['te']
FEATS = F + TE


def to_num(df):
    x = df[FEATS].copy()
    for c in ['route_id', 'seg_id']:
        x[c] = x[c].astype(G['CAT_DTYPE'][c]).cat.codes
    return x.astype(float).fillna(-1).values


MODELS = {
    'RandomForest': lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.5, n_jobs=-1, random_state=0),
    'ExtraTrees': lambda: ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.7, n_jobs=-1, random_state=0),
}
t0 = time.time()
folds = []
for dd in DATES:
    fm = (TR.date != dd).values
    f, v = G['add_te'](TR[fm], TR[~fm])
    folds.append((to_num(f), np.log(f.tt_sec.values), to_num(v), v.index))
print(f'fold 준비 {time.time() - t0:.0f}초 | 피처 {len(FEATS)}개', flush=True)

old = pd.read_parquet(f'{ROOT}/experiments/oof_final_E41.parquet').set_index('row_id').loc[TR.row_id.values]
base = pd.Series(old.e41.values, index=TR.index)
oofs = {}
for name, mk in MODELS.items():
    o = pd.Series(np.nan, index=TR.index)
    for xf, yf, xv, iv in folds:
        o[iv] = np.exp(mk().fit(xf, yf).predict(xv))
    oofs[name] = o
    print(f'{name} 끝 ({time.time() - t0:.0f}초)', flush=True)

K = ['월', '화', '수', '토', '예상']
r0 = cv_of(base)
rows = [{'실험': 'E41 섞기 (LGB 0.8 + CB 0.2)', **{k: r0[k] for k in K}}]
best = None
for name, o in oofs.items():
    r = cv_of(o); rows.append({'실험': f'{name} 혼자', **{k: r[k] for k in K}})
    for w in [0.1, 0.2, 0.3]:
        r = cv_of((1 - w) * base + w * o)
        wd = sum(r[d] < r0[d] for d in ['월', '화', '수'])
        rows.append({'실험': f'E41 섞기 + {name} {w}', **{k: r[k] for k in K}, '평일 좋아진 날': wd})
        if r['예상'] < r0['예상'] and wd >= 2 and (best is None or r['예상'] < best[2]):
            best = (name, w, r['예상'])
T = pd.DataFrame(rows).set_index('실험').round(3)
print(T.to_string(), flush=True)
print(f'→ {"채택: " + best[0] + " " + str(best[1]) if best else "채택 안 함"}', flush=True)
T.to_csv(f'{ROOT}/experiments/E77_cv_table.csv')
pd.DataFrame({'row_id': TR.row_id.values, **{k: v.values for k, v in oofs.items()}}).to_parquet(f'{ROOT}/experiments/E77_oof.parquet')
json.dump({'best': best}, open(f'{ROOT}/experiments/E77_choice.json', 'w'), ensure_ascii=False)
