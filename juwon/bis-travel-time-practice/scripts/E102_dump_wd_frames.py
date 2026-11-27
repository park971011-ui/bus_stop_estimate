"""E102 준비: 평일 3-fold 검증 행(prep 후 프레임)에서 후처리에 필요한 컬럼만 저장 (모델 학습 없음)"""
from pathlib import Path
import sys
root = Path(__file__).resolve().parents[1]
sys.argv = [sys.argv[0], "final", "MD"]
exec((root / "scripts/E89_wd_lp.py").read_text().split("\nbody = body.replace")[0])
outs = []
for date in sorted(TRAIN.loc[dow(TRAIN) < 5, "date"].unique()):
    valid = TRAIN.date.eq(date)
    _, val = prep(TRAIN[~valid], TRAIN[valid])
    outs.append(val)
    print(date, val.shape, flush=True)
v = pd.concat(outs)
print(v.columns.tolist())
v.to_parquet(root / "experiments/E102_wd_frames.parquet", index=False)
