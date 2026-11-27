"""E107 제출 파일
  목·금 = 0.9×(0.5×(0.9×LGB잎511·mc5 + 0.1×CB) + 0.5×평일LGB잎255·mc5) + 0.1×ET  → 시간 합 후처리 frac 0.25
  일요일 = E106 그대로 (E104 평균 + 시간 합 후처리 frac 0.4)
  평일 3-fold (후처리 포함): E105 23.056 → 22.978 (3/3)
"""
from pathlib import Path
import sys
root = Path(__file__).resolve().parents[1]
sys.argv = [sys.argv[0], "final", "MD"]
exec((root / "scripts/E89_wd_lp.py").read_text().split("\nbody = body.replace")[0])
from hour_sum_adjust import hour_sum_adjust
D = root / 'experiments/E103_oof'
r = lambda n: pd.read_parquet(D / f'{n}_final.parquet').set_index('row_id').pred
L5, W5, CBp = r('L_511mc5'), r('W_255mc5'), r('C_base')
ids = L5.index
e67 = pd.read_csv(root / 'submissions/final_E67_sunday_weight_route.csv').set_index('row_id').tt_sec
e101 = pd.read_csv(root / 'submissions/final_E101_E67_et_mae20.csv').set_index('row_id').tt_sec
et = (e101.loc[ids] - 0.8 * e67.loc[ids]) / 0.2
wd = 0.9 * (0.5 * (0.9 * L5 + 0.1 * CBp.loc[ids]) + 0.5 * W5.loc[ids]) + 0.1 * et
_, te = prep(TRAIN, TEST)
tw = te[dow(te) < 5].reset_index(drop=True)
p = wd.loc[tw.row_id].values
a, nb = hour_sum_adjust(tw, p, frac=0.25, turn_share=1.0, pw=1.0)
e106 = pd.read_csv(root / 'submissions/final_E106_E105_hoursum.csv').set_index('row_id').tt_sec
new = e106.copy(); new.loc[tw.row_id] = np.round(np.maximum(a, 0), 1)
d = new.loc[tw.row_id].values - e106.loc[tw.row_id].values
print(f'목·금 block {nb} | E106 대비 평균 {d.mean():+.2f}초, |차이| {np.abs(d).mean():.2f}초')
ss = pd.read_csv(root / 'data/final/sample_submission.csv')
o = new.loc[ss.row_id].reset_index(); o.columns = ['row_id', 'tt_sec']
assert len(o) == 111563 and o.tt_sec.notna().all() and (o.tt_sec >= 0).all() and not o.row_id.duplicated().any()
path = root / 'submissions/final_E107_mc5_hoursum.csv'; assert not path.exists()
o.to_csv(path, index=False); print('저장', path)
