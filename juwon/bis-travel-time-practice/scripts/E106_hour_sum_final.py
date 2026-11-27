"""E106: E105 제출 파일 + 시간 합 후처리 (scripts/hour_sum_adjust.py, 회차 행에 차이의 frac를 나눔)
  평일 3-fold (E105 평일 섞기): frac 0.25 23.056 → 23.010 (-0.046, 3/3)
  연습 일요일 노선 빼기 (E104 평균): frac 0.25 -0.19, frac 0.4 -0.26 (둘 다 4/4)
  → 목·금 frac 0.25, 일요일 frac 0.4, turn_share 1, pw 1
"""
from pathlib import Path
import sys, os
root = Path(__file__).resolve().parents[1]
sys.argv = [sys.argv[0], "final", "MD"]
exec((root / "scripts/E89_wd_lp.py").read_text().split("\nbody = body.replace")[0])
from hour_sum_adjust import hour_sum_adjust
_, te = prep(TRAIN, TEST)                         # 회차 구간·trip_no 붙은 test 전체 (예측엔 안 씀)
base = pd.read_csv(root / 'submissions/final_E105_bigtree_et_wd_e104sun.csv').set_index('row_id').tt_sec
te = te.reset_index(drop=True)
p = base.loc[te.row_id].values
sun = (dow(te) == 6)
out = p.copy()
for mask, frac in [(~sun, 0.25), (sun, 0.4)]:
    sub = te[mask].reset_index(drop=True)
    a, nb = hour_sum_adjust(sub, p[mask], frac=frac, turn_share=1.0, pw=1.0)
    out[mask] = a
    dd = a - p[mask]
    print(f"{'일' if frac == 0.4 else '목·금'}: block {nb}, 바뀐 행 {(dd != 0).sum()}, 평균 {dd.mean():+.3f}초, |차이| {np.abs(dd).mean():.3f}초, 최대 {np.abs(dd).max():.0f}초", flush=True)
new = base.copy(); new.loc[te.row_id] = np.round(out, 1)
ss = pd.read_csv(root / 'data/final/sample_submission.csv')
o = new.loc[ss.row_id].reset_index(); o.columns = ['row_id', 'tt_sec']
assert len(o) == 111563 and o.tt_sec.notna().all() and (o.tt_sec >= 0).all() and not o.row_id.duplicated().any()
path = root / 'submissions/final_E106_E105_hoursum.csv'; assert not path.exists()
o.to_csv(path, index=False); print('저장', path)
