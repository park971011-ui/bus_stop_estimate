"""E100: E89 MD + 정류장 공유 다른 노선 버스 수, 평일 날짜별 fold (E91 틀)
  oth_n: 같은 날·같은 시·같은 출발 정류장의 다른 노선 기록 수 / own_n: 같은 노선 기록 수 (대회 행 16.5%가 2개 노선 공유 정류장)
  (아래는 E98 틀 설명)
  E49는 정각 위치로 대충 잡아서 효과 없었음 → E87 분 복원(lp_tod, 출발 시각 추정 오차 약 2.3분)으로 다시
  hw_prev : 같은 날·노선·방향·출발 정류장에서 바로 앞 차가 몇 분 먼저 출발했나 (다른 차량만)
  hw_next : 바로 뒤 차가 몇 분 뒤 출발하나
  hw_ratio: hw_prev / 그날 배차 간격 (1보다 작으면 몰림, 크면 벌어짐)
  입력만 사용, 같은 날짜 안에서만 계산. 기준 예측 = E91_weekday_oof.parquet baseline (23.2068)
실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E98_headway_lp.py
"""
from pathlib import Path
import sys, time

root = Path(__file__).resolve().parents[1]
sys.argv = [sys.argv[0], "final", "MD"]
exec((root / "scripts/E89_wd_lp.py").read_text().split("\nbody = body.replace")[0])
output = root / "experiments"
HW = ["oth_n", "own_n"]


def add_hw(df):
    c = df.groupby(["date", "arrive_hour", "from_stop", "route_id"]).size().rename("n").reset_index()
    tot = c.groupby(["date", "arrive_hour", "from_stop"]).n.transform("sum")
    c["oth_n"] = tot - c.n; c["own_n"] = c.n
    m = df[["date", "arrive_hour", "from_stop", "route_id"]].merge(c, how="left")
    return df.assign(oth_n=m.oth_n.values, own_n=m.own_n.values)


def predict(fit, val, fw, vw, cols):
    l_model = lgb.LGBMRegressor(**LGB).fit(to_X(fit, cols), fit.tt_sec)
    c_model = CatBoostRegressor(**CB, allow_writing_files=False, cat_features=["route_id", "seg_id"]).fit(fit[cols], fit.tt_sec)
    w_model = lgb.LGBMRegressor(**LGB).fit(to_X(fw, cols), fw.tt_sec)
    mixed = .8 * l_model.predict(to_X(val, cols)) + .2 * c_model.predict(val[cols])
    return .5 * mixed + .5 * w_model.predict(to_X(vw, cols))


t0 = time.time()
base = pd.read_parquet(output / "E91_weekday_oof.parquet")[["row_id", "baseline"]]
folds = []
for D in sorted(TRAIN.loc[dow(TRAIN) < 5, "date"].unique()):
    m = TRAIN.date.eq(D)
    fit, val = map(add_hw, prep(TRAIN[~m], TRAIN[m]))
    fw, vw = map(add_hw, prep(TRAIN[~m & (dow(TRAIN) < 5)], TRAIN[m]))
    assert val.row_id.equals(vw.row_id)
    res = val[["row_id", "date", "tt_sec", "is_turn2"] + HW].copy()
    res["E100"] = predict(fit, val, fw, vw, FE + HW)
    res = res.merge(base, on="row_id")
    print(f"{D} 공유 정류장 행 {(res.oth_n > 0).mean():.1%} | E89 {(res.baseline - res.tt_sec).abs().mean():.4f} → E100 {(res.E100 - res.tt_sec).abs().mean():.4f} ({time.time() - t0:.0f}초)", flush=True)
    folds.append(res); pd.concat(folds).to_parquet(output / "E100_weekday_oof.parquet", index=False)
rows = pd.concat(folds); assert len(rows) == 120769
turn = rows.is_turn2.eq(1)
out = []
for g, sub in {**{d: rows[rows.date == d] for d in sorted(rows.date.unique())}, "평일 전체": rows, "일반 구간": rows[~turn], "회차 구간": rows[turn]}.items():
    b, c = (sub.baseline - sub.tt_sec).abs().mean(), (sub.E100 - sub.tt_sec).abs().mean()
    out.append({"group": g, "n": len(sub), "E89": b, "E100": c, "improvement": b - c})
t = pd.DataFrame(out); t.to_csv(output / "E100_cv.csv", index=False)
print(t.round(4).to_string(index=False))
print(rows.groupby(rows.oth_n.gt(0)).apply(lambda x: pd.Series({"n": len(x), "E89": (x.baseline - x.tt_sec).abs().mean(), "E100": (x.E100 - x.tt_sec).abs().mean()})).round(3).to_string())
