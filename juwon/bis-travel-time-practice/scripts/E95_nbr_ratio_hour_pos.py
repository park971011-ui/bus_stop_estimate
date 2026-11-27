"""E95: E89 MD + 피처 A(주변 구간 시간대 혼잡비) / B(정각 위치차 km) — 평일 일반 구간 개선 목표
  A nbr_ratio  : 같은 노선·방향에서 stop_seq ±1, ±2 이웃(현재·회차 구간 제외, 방향 넘어가지 않음)마다
                 (그 시 평일 tt 중앙값 ÷ 그 이웃 평일 전체 중앙값). 이웃별 그 시 관측 3건 이상, 유효 이웃 2개 이상 → 비율들의 중앙값
  B pos_diff_km: 같은 운행에서 연속 정류장(stop_seq +1)의 arrive_hour가 h → h+1이면, 처음 h+1로 기록된 행의 출발 정류장 누적거리(km)
                 − 같은 노선·방향·운행 시작 정류장·운행 시작 시·정각 시를 가진 다른 날짜 평일 운행들의 위치 중앙값 (3건 이상)
                 경계 행부터 같은 운행·같은 시 행에만 값. 기록이 끊긴 경계, 비교 3건 미만, 경계 전 행은 결측
  두 피처 모두 기준 통계 = 학습(fit) 데이터 중 평일이면서 그 행 날짜가 아닌 날짜. 토·일 행은 결측.
  기준 모델·시드·하이퍼파라미터·섞는 비율은 E89 MD 그대로 (E91 틀). 기준 예측은 E91_weekday_oof.parquet의 baseline 재사용.
실행: cd bis-travel-time-practice && /opt/miniconda3/envs/bis/bin/python scripts/E95_nbr_ratio_hour_pos.py cv
      (통과 후보가 있으면) ... E95_nbr_ratio_hour_pos.py practice A|B|AB
"""
from pathlib import Path
import sys, time

MODE = sys.argv[1]
CAND_ARG = sys.argv[2] if len(sys.argv) > 2 else None
root = Path(__file__).resolve().parents[1]
sys.argv = [sys.argv[0], "final" if MODE == "cv" else "practice", "MD"]
source = (root / "scripts/E89_wd_lp.py").read_text()
exec(source.split("\nbody = body.replace")[0])          # TRAIN, TEST, FE, prep, LGB, CB, to_X, dow, TRIP ...
output = root / "experiments"
KS = ["route_id", "direction", "stop_seq"]
A, B = "nbr_ratio", "pos_diff_km"
CANDS = {"A": [A], "B": [B], "AB": [A, B]}

# 노선 구조 (입력만): 노선·방향마다 stop_seq는 빈 번호 없이 이어지고 구간 하나에 하나씩 대응 (435개 확인)
_inp = pd.concat([TRAIN, TEST])[KS + ["dist_m"]].drop_duplicates(KS).sort_values(KS)
_inp["cum_km"] = (_inp.groupby(["route_id", "direction"]).dist_m.cumsum() - _inp.dist_m) / 1000   # 출발 정류장 누적거리
CUM = _inp.set_index(KS).cum_km


def ratio_table(S):
    """S(기준 통계 행) → (노선, 방향, stop_seq, 시)별 그 시 중앙값 ÷ 구간 전체 중앙값 (그 시 3건 이상)"""
    allmed = S.groupby(KS).tt_sec.median()
    hh = S.groupby(KS + ["arrive_hour"]).tt_sec.agg(["median", "size"])
    hh = hh[hh["size"] >= 3]
    return hh["median"] / allmed.reindex(hh.index.droplevel(3)).values


def feat_a(rows, S, turn):
    rt = ratio_table(S)
    cols = []
    for off in (-2, -1, 1, 2):
        nb = rows.stop_seq + off
        key = pd.MultiIndex.from_arrays([rows.route_id, rows.direction, nb, rows.arrive_hour])
        v = rt.reindex(key).values                                   # 다른 방향 번호면 (방향, 번호) 키가 없어서 결측
        is_turn = pd.MultiIndex.from_arrays([rows.route_id, rows.direction, nb]).isin(turn)
        cols.append(np.where(is_turn, np.nan, v))
    M = np.column_stack(cols)
    ok = np.isfinite(M).sum(1) >= 2
    out = np.full(len(rows), np.nan)
    out[ok] = np.nanmedian(M[ok], axis=1)
    return out


def events(df):
    """운행마다 정각 경계 사건: (TRIP, 시작 정류장, 시작 시, H, 위치 km, 유효 여부)"""
    s = df.sort_values(TRIP + ["stop_seq"])
    g = s.groupby(TRIP)
    s = s.assign(start_stop=g.from_stop.transform("first"), start_hour=g.arrive_hour.transform("first"),
                 ps=g.stop_seq.shift(1), ph=g.arrive_hour.shift(1))
    e = s[s.arrive_hour.eq(s.ph + 1)].copy()                          # 시가 정확히 1 늘어난 첫 행
    e["valid"] = e.stop_seq.eq(e.ps + 1)                              # 정류장 기록이 이어진 경계만
    e["pos"] = CUM.reindex(pd.MultiIndex.from_arrays([e.route_id, e.direction, e.stop_seq])).values
    e = e.rename(columns={"arrive_hour": "H"})
    return e[TRIP + ["start_stop", "start_hour", "H", "pos", "valid"]]


EK = ["route_id", "direction", "start_stop", "start_hour", "H"]


def feat_b(rows, ev_rows, ev_S):
    ref = ev_S[ev_S.valid].groupby(EK).pos.agg(["median", "size"])
    ref = ref[ref["size"] >= 3]["median"]
    e = ev_rows[ev_rows.valid].copy()
    e["val"] = e.pos.values - ref.reindex(pd.MultiIndex.from_frame(e[EK])).values
    e = e.dropna(subset=["val"]).rename(columns={"H": "arrive_hour"})
    m = rows[TRIP + ["arrive_hour"]].merge(e[TRIP + ["arrive_hour", "val"]], on=TRIP + ["arrive_hour"], how="left")
    return m.val.values


def add_new(df, FIT):
    """df 행마다 기준 통계 = FIT 중 평일이면서 그 행 날짜가 아닌 날짜. 토·일 행은 결측."""
    df = df.copy(); df[A] = np.nan; df[B] = np.nan
    fit_wd = FIT[dow(FIT) < 5]
    turn = pd.MultiIndex.from_frame(FIT.loc[FIT.is_turn2 == 1, KS].drop_duplicates())
    ev_fit, ev_df = events(fit_wd), events(df)
    for D in df.date.unique():
        if pd.Timestamp(D).day_of_week >= 5:
            continue
        m = (df.date == D).values
        S = fit_wd[fit_wd.date != D]
        rows = df[m]
        df.loc[m, A] = feat_a(rows, S, turn)
        df.loc[m, B] = feat_b(rows, ev_df[ev_df.date == D], ev_fit[ev_fit.date != D])
    return df


def predict(fit, val, fw, vw, cols):
    l_model = lgb.LGBMRegressor(**LGB).fit(to_X(fit, cols), fit.tt_sec)
    c_model = CatBoostRegressor(**CB, allow_writing_files=False, cat_features=["route_id", "seg_id"]).fit(fit[cols], fit.tt_sec)
    w_model = lgb.LGBMRegressor(**LGB).fit(to_X(fw, cols), fw.tt_sec)
    mixed = .8 * l_model.predict(to_X(val, cols)) + .2 * c_model.predict(val[cols])
    return .5 * mixed + .5 * w_model.predict(to_X(vw, cols))


t0 = time.time()
if MODE == "cv":
    base = pd.read_parquet(output / "E91_weekday_oof.parquet")[["row_id", "baseline"]].sort_values("row_id")
    assert len(base) == 120769
    folds = []
    for date in sorted(TRAIN.loc[dow(TRAIN) < 5, "date"].unique()):
        valid = TRAIN.date.eq(date)
        fit, val = prep(TRAIN[~valid], TRAIN[valid])
        fw, vw = prep(TRAIN[~valid & (dow(TRAIN) < 5)], TRAIN[valid])
        fit, val, fw, vw = add_new(fit, fit), add_new(val, fit), add_new(fw, fw), add_new(vw, fw)
        assert val.row_id.equals(vw.row_id)
        assert np.allclose(val[[A, B]].values, vw[[A, B]].values, equal_nan=True)   # 두 모델의 검증 피처가 같아야 함
        res = val[["row_id", "date", "route_nm", "tt_sec", "stop_seq", "is_turn2", A, B]].copy()
        print(f"{date} 피처 준비 ({time.time() - t0:.0f}초) | 검증 유효 A {res[A].notna().mean():.1%} B {res[B].notna().mean():.1%}"
              f" | 학습 평일 유효 A {fit.loc[dow(fit) < 5, A].notna().mean():.1%} B {fit.loc[dow(fit) < 5, B].notna().mean():.1%}", flush=True)
        for name, extra in CANDS.items():
            res[name] = predict(fit, val, fw, vw, FE + extra)
            print(f"{date} {name} {(res[name] - res.tt_sec).abs().mean():.4f} ({time.time() - t0:.0f}초)", flush=True)
        folds.append(res)
        pd.concat(folds).to_parquet(output / "E95_weekday_oof.parquet", index=False)
    rows = pd.concat(folds).merge(base, on="row_id", how="left").sort_values("row_id")
    assert rows.row_id.is_unique and len(rows) == 120769 and rows.baseline.notna().all()
    assert abs((rows.baseline - rows.tt_sec).abs().mean() - 23.20676) < 1e-4
    rows.to_parquet(output / "E95_weekday_oof.parquet", index=False)
    turn = rows.is_turn2.eq(1)
    groups = {**{d: rows[rows.date == d] for d in sorted(rows.date.unique())}, "평일 전체": rows, "일반 구간": rows[~turn], "회차 구간": rows[turn]}
    out = []
    for g, sub in groups.items():
        r = {"group": g, "n": len(sub), "valid_A": sub[A].notna().mean(), "valid_B": sub[B].notna().mean(),
             "baseline": (sub.baseline - sub.tt_sec).abs().mean()}
        for name in CANDS:
            r[name] = (sub[name] - sub.tt_sec).abs().mean(); r[f"imp_{name}"] = r["baseline"] - r[name]
        out.append(r)
    table = pd.DataFrame(out)
    table.to_csv(output / "E95_cv.csv", index=False)
    pd.set_option("display.width", 250)
    print(table.round(4).to_string(index=False))
    days = table[table.group.str.startswith("2024")]
    passed = {}
    for name in CANDS:
        tot = table.loc[table.group == "평일 전체", f"imp_{name}"].item()
        gen = table.loc[table.group == "일반 구간", f"imp_{name}"].item()
        nd = int((days[f"imp_{name}"] > 0).sum())
        ok = tot > 0 and gen > 0 and nd >= 2
        print(f"[판정] {name}: 평일 전체 {tot:+.4f}, 일반 {gen:+.4f}, 좋아진 날 {nd}/3 → {'통과' if ok else '탈락'}")
        if ok: passed[name] = table.loc[table.group == "평일 전체", name].item()
    best = min(passed, key=passed.get) if passed else None
    print(f"[판정] 선택: {best or '없음 (E89 유지)'}")
    if best:
        a_ = table.loc[table.group == "평일 전체"]
        bestsingle = min(["A", "B"], key=lambda k: a_[k].item())
        print(f"[참고] A+B {a_['AB'].item():.4f} vs 더 좋은 단일 {bestsingle} {a_[bestsingle].item():.4f}")
else:
    assert CAND_ARG in CANDS
    f, a = prep(TRAIN, TEST)
    fw, aw = prep(TRAIN[dow(TRAIN) < 5], TEST)
    f, a, fw, aw = add_new(f, f), add_new(a, f), add_new(fw, fw), add_new(aw, fw)
    tf = dow(a) < 5                                                  # 연습 진짜 목·금만 채점 (일요일은 보지 않음)
    res = a.loc[tf, ["row_id", "date", "tt_sec", "is_turn2", A, B]].copy()
    print(f"[practice] 목·금 유효 A {res[A].notna().mean():.1%} B {res[B].notna().mean():.1%} ({time.time() - t0:.0f}초)", flush=True)
    for name, cols in [("E89", FE), (CAND_ARG, FE + CANDS[CAND_ARG])]:
        res[name] = predict(f, a, fw, aw, cols)[tf]
        print(f"[practice] {name} 목·금 {(res[name] - res.tt_sec).abs().mean():.4f} ({time.time() - t0:.0f}초)", flush=True)
    res.to_parquet(output / f"E95_practice_{CAND_ARG}.parquet", index=False)
    out = []
    turn = res.is_turn2.eq(1)
    for g, sub in {**{d: res[res.date == d] for d in sorted(res.date.unique())}, "목·금 전체": res, "일반 구간": res[~turn], "회차 구간": res[turn]}.items():
        e0, e1 = (sub.E89 - sub.tt_sec).abs().mean(), (sub[CAND_ARG] - sub.tt_sec).abs().mean()
        out.append({"group": g, "n": len(sub), "E89": e0, CAND_ARG: e1, "improvement": e0 - e1})
    t = pd.DataFrame(out); t.to_csv(output / f"E95_practice_{CAND_ARG}.csv", index=False)
    print(t.round(4).to_string(index=False))
