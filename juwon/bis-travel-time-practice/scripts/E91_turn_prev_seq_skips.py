"""Compare E89 MD with one extra turnaround feature on weekday date folds."""
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.argv = [sys.argv[0], "final", "MD"]
source = (root / "scripts/E89_wd_lp.py").read_text()
exec(source.split("\nbody = body.replace")[0])
output = root / "experiments"
feature = "turn_prev_seq_skips"


def with_skips(frame):
    ordered = frame.sort_values(TRIP + ["stop_seq", "row_id"])
    step = ordered.groupby(TRIP).stop_seq.diff()
    skips = (step - 1).where(ordered.is_turn2.eq(1) & step.ge(1))
    return frame.assign(turn_prev_seq_skips=skips)


def predict_fold(date):
    valid = TRAIN.date.eq(date)
    fit, val = map(with_skips, prep(TRAIN[~valid], TRAIN[valid]))
    fw, vw = map(with_skips, prep(TRAIN[~valid & (dow(TRAIN) < 5)], TRAIN[valid]))
    assert val.row_id.equals(vw.row_id)
    result = val[["row_id", "date", "route_nm", "tt_sec", "stop_seq", "is_turn2", feature]].copy()
    for name, cols in [("baseline", FE), ("candidate", FE + [feature])]:
        l_model = lgb.LGBMRegressor(**LGB).fit(to_X(fit, cols), fit.tt_sec)
        c_model = CatBoostRegressor(**CB, allow_writing_files=False, cat_features=["route_id", "seg_id"]).fit(fit[cols], fit.tt_sec)
        w_model = lgb.LGBMRegressor(**LGB).fit(to_X(fw, cols), fw.tt_sec)
        mixed = .8 * l_model.predict(to_X(val, cols)) + .2 * c_model.predict(val[cols])
        result[name] = .5 * mixed + .5 * w_model.predict(to_X(vw, cols))
        print(date, name, (result[name] - result.tt_sec).abs().mean(), flush=True)
    return result


folds = []
for date in sorted(TRAIN.loc[dow(TRAIN) < 5, "date"].unique()):
    folds.append(predict_fold(date))
    pd.concat(folds).to_parquet(output / "E91_weekday_oof.parquet", index=False)
rows = pd.concat(folds)
assert rows.row_id.is_unique and len(rows) == 120769
assert np.isfinite(rows[["baseline", "candidate"]].values).all()
groups = {"전체": rows, **dict(tuple(rows.groupby("date")))}
turn = rows.is_turn2.eq(1)
groups.update({"회차": rows[turn], "일반": rows[~turn]})
for skip in (0, 2):
    groups[f"704_75_skip{skip}"] = rows[turn & rows.route_nm.astype(str).eq("704") & rows.stop_seq.eq(75) & rows[feature].eq(skip)]
summary = []
for name, sub in groups.items():
    base = (sub.baseline - sub.tt_sec).abs().mean()
    new = (sub.candidate - sub.tt_sec).abs().mean()
    summary.append((name, len(sub), base, new, base - new))
table = pd.DataFrame(summary, columns=["group", "n", "baseline", "candidate", "improvement"])
table.to_csv(output / "E91_cv.csv", index=False)
print(table.to_string(index=False))
