"""Build E37: route-held-out adaptive Sunday blend.

The two Sunday counterfactuals already used by E35 are treated as endpoints:
"Saturday-like" (E32) and "weekday-like" (recovered from the 50/50 E35
blend).  A small LightGBM gate learns, from the provided practice routes,
how far to move between the endpoints.  It never extrapolates outside them.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUB = ROOT / "submissions"

FEATURES = [
    "sat",
    "wd",
    "delta",
    "absdelta",
    "dist_m",
    "arrive_hour",
    "direction",
    "route_type",
    "stop_seq",
    "interval_wd",
    "interval_sat",
    "interval_sun",
    "from_lat",
    "from_lon",
    "to_lat",
    "to_lon",
    "pred_per_m",
]

CONFIGS = [
    {"num_leaves": 7, "min_child_samples": 100},
    {"num_leaves": 15, "min_child_samples": 100},
    {"num_leaves": 31, "min_child_samples": 100},
]
CORRECTION_CAP = 150.0


def add_counterfactuals(data, e32_path, e35_path):
    e32 = pd.read_csv(e32_path).rename(columns={"tt_sec": "sat"})
    e35 = pd.read_csv(e35_path).rename(columns={"tt_sec": "mix"})
    out = data.merge(e32, on="row_id", validate="one_to_one")
    out = out.merge(e35, on="row_id", validate="one_to_one")
    out["wd"] = 2 * out["mix"] - out["sat"]
    out["delta"] = out["wd"] - out["sat"]
    out["absdelta"] = out["delta"].abs()
    out["pred_per_m"] = out["sat"] / out["dist_m"].clip(lower=1)
    return out


def sunday_only(df):
    return df[pd.to_datetime(df["date"]).dt.day_of_week == 6].copy()


def fit_gate(train, config):
    params = {
        "objective": "regression_l1",
        "n_estimators": 200,
        "learning_rate": 0.03,
        "reg_lambda": 10,
        "random_state": 0,
        "verbose": -1,
        "n_jobs": -1,
        **config,
    }
    return lgb.LGBMRegressor(**params).fit(train[FEATURES], train["target_res"])


def clipped_correction(df, raw_correction):
    lower = np.minimum(0, df["delta"].to_numpy())
    upper = np.maximum(0, df["delta"].to_numpy())
    correction = np.clip(raw_correction, lower, upper)
    return np.clip(correction, -CORRECTION_CAP, CORRECTION_CAP)


def route_held_out_prediction(practice_sun):
    predictions = []
    for config in CONFIGS:
        pred = pd.Series(index=practice_sun.index, dtype=float)
        for route in practice_sun["route_nm"].unique():
            fit = practice_sun[practice_sun["route_nm"] != route]
            val = practice_sun[practice_sun["route_nm"] == route]
            pred.loc[val.index] = fit_gate(fit, config).predict(val[FEATURES])
        predictions.append(pred)
    raw = pd.concat(predictions, axis=1).mean(axis=1).to_numpy()
    return clipped_correction(practice_sun, raw)


def main():
    practice = pd.read_parquet(ROOT / "data/practice/test.parquet")
    final = pd.read_parquet(ROOT / "data/final/test.parquet")

    practice = add_counterfactuals(
        practice,
        SUB / "practice_E32.csv",
        SUB / "practice_E35_sunmix.csv",
    )
    final = add_counterfactuals(
        final,
        SUB / "final_E32.csv",
        SUB / "final_E35_sunmix.csv",
    )

    practice_sun = sunday_only(practice)
    practice_sun["target_res"] = practice_sun["tt_sec"] - practice_sun["sat"]
    oof_correction = route_held_out_prediction(practice_sun)
    base_mae = np.abs(practice_sun["tt_sec"] - practice_sun["sat"]).mean()
    adaptive_mae = np.abs(
        practice_sun["tt_sec"] - (practice_sun["sat"] + oof_correction)
    ).mean()
    print(f"Practice Sunday route-held-out MAE: {base_mae:.6f} -> {adaptive_mae:.6f}")

    final_sun = sunday_only(final)
    raw_predictions = [
        fit_gate(practice_sun, config).predict(final_sun[FEATURES])
        for config in CONFIGS
    ]
    correction = clipped_correction(final_sun, np.mean(raw_predictions, axis=0))

    prediction = final["sat"].copy()
    prediction.loc[final_sun.index] = final_sun["sat"] + correction
    output = pd.DataFrame({"row_id": final["row_id"], "tt_sec": prediction.round(1)})
    output = output.set_index("row_id").loc[
        pd.read_parquet(ROOT / "data/final/test.parquet")["row_id"]
    ].reset_index()

    if output["tt_sec"].isna().any() or output["row_id"].duplicated().any():
        raise RuntimeError("Invalid submission: missing prediction or duplicate row_id")
    if (output["tt_sec"] < 0).any():
        raise RuntimeError("Invalid submission: negative travel-time prediction")

    path = SUB / "final_E37_adaptive_sunmix.csv"
    output.to_csv(path, index=False)
    changed = int((output["tt_sec"].to_numpy() != final["sat"].round(1).to_numpy()).sum())
    print(f"Saved {path} ({len(output):,} rows, {changed:,} changed vs E32)")
    print(
        "Final Sunday correction quantiles:",
        np.quantile(correction, [0, 0.1, 0.5, 0.9, 1]).round(3).tolist(),
    )


if __name__ == "__main__":
    main()
