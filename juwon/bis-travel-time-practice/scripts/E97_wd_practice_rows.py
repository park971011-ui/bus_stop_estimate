"""E97 목·금 학습에 실습용 데이터 추가 (실습용 데이터 학습 사용은 11/24 확인받음, E59 참고)
  아이디어: 실습용 test 목·금(9/26·27)은 대회 test 목·금과 같은 날짜, 정답 있음 → 그날 도로 상황을 배움 (일요일 E58과 같은 방식)
  모델: E89 MD 그대로 (0.5 × (LGB 0.8 + CB 0.2, 전체 요일) + 0.5 × 평일 전용 LGB, 피처 FE, 시드·설정 같음)
  추가: 학습 = 대회 train + 실습 train + 실습 test 목·금 / 요일 표시 is_mon~is_fri 5개
        피처(회차 힌트·추정 회차·LP 등)는 데이터(대회/실습)별로 따로 계산 (차량·날짜 키가 섞이지 않게)
  검증(대리): 대회 train 평일 날짜 D 하나를 빼고, 실습용의 D 날짜 행은 학습에 남김 → 대회 D 예측
             (test 때 목·금이 실습용에만 있는 상황과 같음). 기준 = E91_weekday_oof baseline (E89 MD, 23.2068)
  일요일은 E89 MD 파일 그대로 (E73).
실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E97_wd_practice_rows.py make   (또는 cv)
"""
from pathlib import Path
import os, sys, time

MODE = sys.argv[1]
root = Path(__file__).resolve().parents[1]
head = (root / "scripts/E89_wd_lp.py").read_text().split("\nbody = body.replace")[0]
sys.argv = [sys.argv[0], "practice", "MD"]; exec(head)
PTR, PTE = TRAIN.copy(), TEST.copy()
sys.argv = [sys.argv[0], "final", "MD"]; exec(head)
FTR, FTE = TRAIN.copy(), TEST.copy()
output = root / "experiments"
DAYF = ["is_mon", "is_tue", "is_wed", "is_thu", "is_fri"]
FX = FE + DAYF
CATS = {c: pd.CategoricalDtype(sorted(pd.concat([PTR[c], PTE[c], FTR[c], FTE[c]]).astype(str).unique())) for c in ["route_id", "seg_id"]}
print(f"실습 train 날짜 {sorted(PTR.date.unique())} | 실습 test 날짜 {sorted(PTE.date.unique())}", flush=True)


def Xf(df, cols):
    x = df[cols].copy()
    for c in CATS:
        if c in x: x[c] = x[c].astype(str).astype(CATS[c])
    return x


def flags(df):
    df = df.reset_index(drop=True)
    d = dow(df)
    for i, c in enumerate(DAYF): df[c] = (d == i).astype(int)
    return df


def prep_fit(raw):
    """학습용 행만 준비 (prep은 적용 대상이 필요해서 작은 조각을 넣고 버림)"""
    return prep(raw, raw.iloc[:5])[0]


P = pd.concat([PTR, PTE[dow(PTE) < 5]], ignore_index=True)               # 실습 train + 실습 test 목·금 (일요일 제외)
t0 = time.time()
fp, fwp = flags(prep_fit(P)), flags(prep_fit(P[dow(P) < 5].reset_index(drop=True)))
assert len(fp) == len(P) and fp.row_id.is_unique, "실습 행 중복"
print(f"실습 행 준비 {len(fp):,} (평일 {len(fwp):,}) ({time.time() - t0:.0f}초)", flush=True)


def predict(fit, val, fw, vw, cols):
    l_model = lgb.LGBMRegressor(**LGB).fit(Xf(fit, cols), fit.tt_sec)
    c_model = CatBoostRegressor(**CB, allow_writing_files=False, cat_features=["route_id", "seg_id"])
    c_model.fit(fit[cols].astype({"route_id": str, "seg_id": str}), fit.tt_sec)
    w_model = lgb.LGBMRegressor(**LGB).fit(Xf(fw, cols), fw.tt_sec)
    mixed = .8 * l_model.predict(Xf(val, cols)) + .2 * c_model.predict(val[cols].astype({"route_id": str, "seg_id": str}))
    return .5 * mixed + .5 * w_model.predict(Xf(vw, cols))


if MODE == "make":
    f, a = prep(FTR, FTE)
    fw, aw = prep(FTR[dow(FTR) < 5], FTE)
    f, a, fw, aw = flags(f), flags(a), flags(fw), flags(aw)
    assert a.row_id.equals(aw.row_id)
    p = predict(pd.concat([f, fp], ignore_index=True), a, pd.concat([fw, fwp], ignore_index=True), aw, FX)
    tf = dow(a) < 5
    pd.DataFrame({'row_id': a.row_id.values, 'raw': p}).to_parquet(output / 'E97_test_raw.parquet', index=False)
    print(f'목·금 예측 최솟값 {p[tf].min():.2f}, 음수 {(p[tf] < 0).sum()}행, 결측 {np.isnan(p[tf]).sum()}행 → 음수는 0으로', flush=True)
    p = np.clip(p, 0, None)
    e89 = pd.read_csv(root / "submissions/final_E89_wd_lpMD_E73sun.csv").set_index("row_id").tt_sec
    new = e89.copy(); new.loc[a.row_id.values[tf]] = np.round(p[tf], 1)
    ss = pd.read_csv(root / "data/final/sample_submission.csv")
    sub = new.loc[ss.row_id].reset_index(); sub.columns = ["row_id", "tt_sec"]
    assert len(sub) == 111563 and sub.tt_sec.notna().all() and not sub.row_id.duplicated().any() and (sub.tt_sec >= 0).all()
    path = root / "submissions/final_E97_wd_practice_E73sun.csv"; assert not path.exists(); sub.to_csv(path, index=False)
    ids = a.row_id.values[tf]; d = new.loc[ids].values - e89.loc[ids].values
    dd = pd.Series(d).groupby(pd.Series(dow(a)[tf])).agg(lambda x: x.mean())
    print(f"저장 {path} | 목·금 E89 대비 평균 {d.mean():+.2f}초, |차이| {np.abs(d).mean():.2f}초 | 요일별 평균 {dd.round(2).to_dict()} ({time.time() - t0:.0f}초)", flush=True)
else:
    base = pd.read_parquet(output / "E91_weekday_oof.parquet")[["row_id", "baseline"]]
    folds = []
    for D in sorted(FTR.loc[dow(FTR) < 5, "date"].unique()):
        m = FTR.date.eq(D)
        f, v = prep(FTR[~m], FTR[m])
        fw, vw = prep(FTR[~m & (dow(FTR) < 5)], FTR[m])
        f, v, fw, vw = flags(f), flags(v), flags(fw), flags(vw)
        res = v[["row_id", "date", "tt_sec", "is_turn2"]].copy()
        res["E97"] = predict(pd.concat([f, fp], ignore_index=True), v, pd.concat([fw, fwp], ignore_index=True), vw, FX)
        res = res.merge(base, on="row_id")
        print(f"{D} E89 {(res.baseline - res.tt_sec).abs().mean():.4f} → E97 {(res.E97 - res.tt_sec).abs().mean():.4f} ({time.time() - t0:.0f}초)", flush=True)
        folds.append(res); pd.concat(folds).to_parquet(output / "E97_weekday_oof.parquet", index=False)
    rows = pd.concat(folds); assert len(rows) == 120769
    turn = rows.is_turn2.eq(1)
    out = []
    for g, sub in {**{d: rows[rows.date == d] for d in sorted(rows.date.unique())}, "평일 전체": rows, "일반 구간": rows[~turn], "회차 구간": rows[turn]}.items():
        b, c = (sub.baseline - sub.tt_sec).abs().mean(), (sub.E97 - sub.tt_sec).abs().mean()
        out.append({"group": g, "n": len(sub), "E89": b, "E97": c, "improvement": b - c})
    t = pd.DataFrame(out); t.to_csv(output / "E97_cv.csv", index=False)
    print(t.round(4).to_string(index=False))
