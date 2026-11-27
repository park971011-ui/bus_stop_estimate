"""E92: E89 MD + 다음 행 순번 건너뜀(next_seq_skips), 평일 날짜별 fold 비교 (E91과 같은 틀).
next_seq_skips = 같은 운행(TRIP)에서 다음 행 stop_seq - 지금 stop_seq - 1. 운행 마지막 행은 결측.
가설: 회차 직전에는 순번이 자주 뛰고(20%), 뛰었을 때 tt 중앙값 566초 vs 87초.
"""
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.argv = [sys.argv[0], "final", "MD"]
source = (root / "scripts/E89_wd_lp.py").read_text()
exec(source.split("\nbody = body.replace")[0])
output = root / "experiments"
feature = "next_seq_skips"


def with_skips(frame):
    ordered = frame.sort_values(TRIP + ["stop_seq", "row_id"])
    step = -ordered.groupby(TRIP).stop_seq.diff(-1)
    return frame.assign(next_seq_skips=(step - 1).where(step.ge(1)))


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
    pd.concat(folds).to_parquet(output / "E92_weekday_oof.parquet", index=False)
rows = pd.concat(folds)
assert rows.row_id.is_unique and len(rows) == 120769
groups = {"전체": rows, **dict(tuple(rows.groupby("date")))}
turn = rows.is_turn2.eq(1)
jump = rows[feature].gt(0)
groups.update({"회차": rows[turn], "일반": rows[~turn], "다음행 뜀": rows[jump], "다음행 안뜀": rows[~jump]})
summary = []
for name, sub in groups.items():
    base = (sub.baseline - sub.tt_sec).abs().mean()
    new = (sub.candidate - sub.tt_sec).abs().mean()
    summary.append((name, len(sub), base, new, base - new))
table = pd.DataFrame(summary, columns=["group", "n", "baseline", "candidate", "improvement"])
table.to_csv(output / "E92_cv.csv", index=False)
print(table.to_string(index=False))
