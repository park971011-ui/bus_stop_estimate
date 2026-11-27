"""E105 제출 파일: 목·금 = 0.5×(0.9×LGB잎511 + 0.1×CB) + 0.25×평일LGB잎255 + 0.25×평일LGB col0.5 (E103 평일 3-fold 23.207→23.068)
  일요일: 인자 'e89' = E89 그대로(E73) / 'e104' = E104 일요일(LGB 기본·잎511 평균)
"""
import sys, os, numpy as np, pandas as pd
ROOT = '/Users/kimjuwon/Desktop/에모팀플/bis-travel-time-practice'
SUN = sys.argv[1]
D = ROOT + '/experiments/E103_oof/'
r = lambda n: pd.read_parquet(D + n + '_final.parquet').set_index('row_id').pred
L5, W2, Wc, C = r('L_nl511'), r('W_nl255'), r('W_col5'), r('C_base')
wd = 0.5 * (0.9 * L5 + 0.1 * C.loc[L5.index]) + 0.25 * W2.loc[L5.index] + 0.25 * Wc.loc[L5.index]
e89 = pd.read_csv(ROOT + '/submissions/final_E89_wd_lpMD_E73sun.csv').set_index('row_id').tt_sec
# ET(E101, 평일 전용 ExtraTrees MAE) 목·금 예측 복원: E101 = 0.8×E67 + 0.2×ET  → 평일 검증 +ET 0.1: 23.068→23.056
e67 = pd.read_csv(ROOT + '/submissions/final_E67_sunday_weight_route.csv').set_index('row_id').tt_sec
e101 = pd.read_csv(ROOT + '/submissions/final_E101_E67_et_mae20.csv').set_index('row_id').tt_sec
et = (e101.loc[wd.index] - 0.8 * e67.loc[wd.index]) / 0.2
print(f'ET 복원: 평균 {et.mean():.2f}, 최소 {et.min():.1f}')
wd = 0.9 * wd + 0.1 * et
sub = e89.copy()
d = wd.values - e89.loc[wd.index].values
print(f'목·금 {len(wd)}행 | E89 대비 평균 {d.mean():+.2f}초, |차이| {np.abs(d).mean():.2f}초, 상관 {np.corrcoef(wd.values, e89.loc[wd.index].values)[0,1]:.4f}')
sub.loc[wd.index] = np.round(np.maximum(wd.values, 0), 1)
if SUN == 'e104':
    s = pd.read_parquet(ROOT + '/experiments/E104_sunday_pred.parquet').set_index('row_id').tt_sec
    ds = s.values - e89.loc[s.index].values
    print(f'일요일 {len(s)}행 | E89 대비 평균 {ds.mean():+.2f}초, |차이| {np.abs(ds).mean():.2f}초')
    sub.loc[s.index] = np.round(np.maximum(s.values, 0), 1)
ss = pd.read_csv(ROOT + '/data/final/sample_submission.csv')
out = sub.loc[ss.row_id].reset_index(); out.columns = ['row_id', 'tt_sec']
assert len(out) == 111563 and out.tt_sec.notna().all() and (out.tt_sec >= 0).all() and not out.row_id.duplicated().any()
path = f'{ROOT}/submissions/final_E105_bigtree_et_wd_{SUN}sun.csv'; assert not os.path.exists(path)
out.to_csv(path, index=False); print('저장', path)
