"""Check adaptive counterfactual gating on the final student training routes.

Because final/train has no Sunday labels, Saturday is used as an analogue:
train the gate on practice Saturday and evaluate it on final Saturday.  Both
base models are trained without Saturday, so every evaluated prediction is
out-of-fold by date.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from E37_adaptive_sunmix import CONFIGS, FEATURES, clipped_correction, fit_gate


ROOT = Path(__file__).resolve().parents[1]
PEAK = {7, 8, 9, 17, 18, 19}
MODEL_FEATURES = [
    "route_id",
    "route_type",
    "direction",
    "interval",
    "from_lat",
    "from_lon",
    "to_lat",
    "to_lon",
    "dlat",
    "dlon",
    "stop_seq",
    "dist_m",
    "arrive_hour",
    "is_weekend",
    "is_peak",
    "seg_id",
    "first_min",
    "last_min",
    "hour_from_first",
    "is_last_seg",
    "seq_from_end",
    "trip_no",
    "seq_in_trip",
    "trip_start_hour",
    "hours_in_service",
]

LGB_PARAMS = {
    "objective": "regression_l1",
    "learning_rate": 0.02,
    "num_leaves": 127,
    "min_child_samples": 10,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "n_estimators": 1500,
    "random_state": 0,
    "verbose": -1,
    "n_jobs": -1,
}
CB_PARAMS = {
    "loss_function": "MAE",
    "iterations": 2000,
    "learning_rate": 0.08,
    "depth": 10,
    "l2_leaf_reg": 3,
    "random_seed": 0,
    "verbose": 0,
    "thread_count": -1,
    "allow_writing_files": False,
}


def hhmm_to_min(x):
    return (x // 100) * 60 + x % 100


def prepare(train_raw, test_raw):
    both = pd.concat([train_raw, test_raw], ignore_index=True)
    last_seq = both.groupby(["route_id", "direction"]).stop_seq.max().rename("last_seq")

    def transform(df):
        df = df.copy()
        dow = pd.to_datetime(df["date"]).dt.day_of_week
        df["is_weekend"] = (dow >= 5).astype(int)
        df["is_peak"] = df["arrive_hour"].isin(PEAK).astype(int)
        df["interval"] = np.select(
            [dow == 5, dow == 6],
            [df["interval_sat"], df["interval_sun"]],
            default=df["interval_wd"],
        )
        df["dlat"] = df["to_lat"] - df["from_lat"]
        df["dlon"] = df["to_lon"] - df["from_lon"]
        df["seg_id"] = df["from_stop"] + "_" + df["to_stop"]
        df["first_min"] = hhmm_to_min(df["first_time"])
        df["last_min"] = hhmm_to_min(df["last_time"])
        df["hour_from_first"] = df["arrive_hour"] * 60 - df["first_min"]
        df = df.join(last_seq, on=["route_id", "direction"])
        df["is_last_seg"] = (df["stop_seq"] == df["last_seq"]).astype(int)
        df["seq_from_end"] = df["last_seq"] - df["stop_seq"]

        key = ["date", "vehicle_id", "route_id", "direction"]
        df = df.sort_values(key + ["arrive_hour", "stop_seq"])
        new_trip = df.groupby(key).stop_seq.diff().fillna(-1) <= 0
        df["trip_no"] = new_trip.groupby([df[k] for k in key]).cumsum()
        trip = key + ["trip_no"]
        df["trip_start_seq"] = df.groupby(trip).stop_seq.transform("min")
        df["seq_in_trip"] = df["stop_seq"] - df["trip_start_seq"]
        df["trip_start_hour"] = df.groupby(trip).arrive_hour.transform("min")
        df["hours_in_service"] = (
            df["arrive_hour"]
            - df.groupby(["date", "vehicle_id"]).arrive_hour.transform("min")
        )
        return df.sort_index()

    train = transform(train_raw)
    test = transform(test_raw)
    categories = {
        col: pd.CategoricalDtype(sorted(pd.concat([train[col], test[col]]).unique()))
        for col in ["route_id", "seg_id"]
    }
    return train, categories


def lightgbm_matrix(df, categories):
    matrix = df[MODEL_FEATURES].copy()
    for col, dtype in categories.items():
        matrix[col] = matrix[col].astype(dtype)
    return matrix


def saturday_counterfactuals(phase):
    train_raw = pd.read_parquet(ROOT / f"data/{phase}/train.parquet")
    test_raw = pd.read_parquet(ROOT / f"data/{phase}/test.parquet")
    train, categories = prepare(train_raw, test_raw)
    saturday = pd.to_datetime(train["date"]).dt.day_of_week == 5
    fit = train[~saturday]
    val_sat = train[saturday].copy()
    val_wd = val_sat.copy()
    val_wd["is_weekend"] = 0
    val_wd["interval"] = val_wd["interval_wd"]

    lgb_model = lgb.LGBMRegressor(**LGB_PARAMS).fit(
        lightgbm_matrix(fit, categories), fit["tt_sec"]
    )
    lgb_sat = lgb_model.predict(lightgbm_matrix(val_sat, categories))
    lgb_wd = lgb_model.predict(lightgbm_matrix(val_wd, categories))

    cb_model = CatBoostRegressor(
        **CB_PARAMS, cat_features=["route_id", "seg_id"]
    ).fit(fit[MODEL_FEATURES], fit["tt_sec"])
    cb_sat = cb_model.predict(val_sat[MODEL_FEATURES])
    cb_wd = cb_model.predict(val_wd[MODEL_FEATURES])

    out = val_sat.copy()
    out["sat"] = 0.8 * lgb_sat + 0.2 * cb_sat
    out["wd"] = 0.8 * lgb_wd + 0.2 * cb_wd
    out["delta"] = out["wd"] - out["sat"]
    out["absdelta"] = out["delta"].abs()
    out["pred_per_m"] = out["sat"] / out["dist_m"].clip(lower=1)
    return out


def mae(y, pred):
    return float(np.abs(np.asarray(y) - np.asarray(pred)).mean())


def main():
    print("Rebuilding Saturday counterfactuals...", flush=True)
    practice = saturday_counterfactuals("practice")
    final = saturday_counterfactuals("final")

    practice["target_res"] = practice["tt_sec"] - practice["sat"]
    raw = np.mean(
        [fit_gate(practice, config).predict(final[FEATURES]) for config in CONFIGS],
        axis=0,
    )
    correction = clipped_correction(final, raw)

    predictions = {
        "E32 Saturday-like": final["sat"],
        "fixed 50:50": final["sat"] + 0.5 * final["delta"],
        "fixed 70:30": final["sat"] + 0.3 * final["delta"],
        "adaptive gate": final["sat"] + correction,
    }
    rows = []
    for name, pred in predictions.items():
        rows.append({"method": name, "n": len(final), "MAE": mae(final["tt_sec"], pred)})
    result = pd.DataFrame(rows)
    result["delta_vs_E32"] = result["MAE"] - result.loc[0, "MAE"]
    print(result.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nRoute-level MAE")
    route_rows = []
    for route, group in final.groupby("route_nm"):
        for name, pred in predictions.items():
            route_rows.append(
                {
                    "route_nm": route,
                    "method": name,
                    "n": len(group),
                    "MAE": mae(group["tt_sec"], pred.loc[group.index]),
                }
            )
    route_result = pd.DataFrame(route_rows)
    print(route_result.pivot(index="route_nm", columns="method", values="MAE").round(3))

    saved = pd.read_parquet(ROOT / "experiments/oof_final_E32.parquet")
    check = final[["row_id"]].merge(saved[["row_id", "e32"]], on="row_id", how="left")
    max_diff = float(np.max(np.abs(check["e32"] - final["sat"].to_numpy())))
    print(f"\nRebuilt E32 vs saved OOF max abs difference: {max_diff:.9f}")

    output = ROOT / "experiments/E37_saturday_check.csv"
    result.to_csv(output, index=False)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
