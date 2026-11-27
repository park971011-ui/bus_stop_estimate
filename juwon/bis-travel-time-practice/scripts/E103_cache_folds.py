"""E103 준비: E89 MD 평일 3-fold의 학습/검증 프레임(prep 후)을 캐시. fit4 = 4일 중 검증일 뺀 3일, fitw = 평일만"""
from pathlib import Path
import sys, pickle
root = Path(__file__).resolve().parents[1]
sys.argv = [sys.argv[0], "final", "MD"]
exec((root / "scripts/E89_wd_lp.py").read_text().split("\nbody = body.replace")[0])
cache = {}
for date in sorted(TRAIN.loc[dow(TRAIN) < 5, "date"].unique()):
    valid = TRAIN.date.eq(date)
    fit, val = prep(TRAIN[~valid], TRAIN[valid])
    fw, vw = prep(TRAIN[~valid & (dow(TRAIN) < 5)], TRAIN[valid])
    cache[date] = dict(fit=fit, val=val, fw=fw, vw=vw)
    print(date, flush=True)
# 최종(목·금) 프레임
fit, te = prep(TRAIN, TEST[dow(TEST) < 5])
fw, tw = prep(TRAIN[dow(TRAIN) < 5], TEST[dow(TEST) < 5])
cache['final'] = dict(fit=fit, val=te, fw=fw, vw=tw)
meta = dict(FE=FE, LGB=LGB, CB=CB, CAT={c: TRAIN[c].astype('category').cat.categories for c in ['route_id', 'seg_id']})
pickle.dump((cache, meta), open(root / 'experiments/E103_folds.pkl', 'wb'))
import inspect; print(inspect.getsource(to_X))
print(LGB)
