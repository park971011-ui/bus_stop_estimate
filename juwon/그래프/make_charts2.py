"""실습 데이터 · 대회 test · 실험 결과 · 제출 파일 그래프 (07~20번)."""
import sys, os, json
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch

ROOT = sys.argv[1]
OUT = f"{ROOT}/그래프"; EXP = f"{ROOT}/bis-travel-time-practice/experiments"
SUB = f"{ROOT}/bis-travel-time-practice/submissions"
os.makedirs(OUT, exist_ok=True)

fm.fontManager.addfont("/System/Library/Fonts/Supplemental/AppleGothic.ttf")
plt.rcParams.update({
    "font.family": "AppleGothic", "axes.unicode_minus": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c9c8c2", "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "text.color": "#0b0b0b",
    "axes.grid": True, "grid.color": "#e8e7e3", "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "figure.dpi": 130, "font.size": 10,
})
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRAY = "#a3a29c"; INK2 = "#52514e"
DOW = ["월", "화", "수", "목", "금", "토", "일"]
S = {}

def save(fig, name):
    fig.tight_layout(); fig.savefig(f"{OUT}/{name}"); plt.close(fig)

def load(p):
    d = pd.read_parquet(p); d["route_nm"] = d["route_nm"].astype(str)
    d["dow"] = pd.to_datetime(d["date"]).dt.dayofweek.map(lambda i: DOW[i]); return d

ptr = load(f"{ROOT}/실습용/train.parquet"); pte = load(f"{ROOT}/실습용/test.parquet")
ftr = load(f"{ROOT}/대회_학생용/train.parquet"); fte = load(f"{ROOT}/대회_학생용/test.parquet")
pall = pd.concat([ptr.assign(part="train"), pte.assign(part="test")])

# 07 실습 날짜별 행 수
c = pall.groupby(["date", "part"]).size().unstack()
fig, ax = plt.subplots(figsize=(8, 3.4)); x = np.arange(len(c))
for part, col in [("train", CAT[0]), ("test", CAT[1])]:
    ax.bar(x, c[part].fillna(0), color=col, width=0.7, label=part)
for i, v in enumerate(c.sum(axis=1)):
    ax.text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=8, color=INK2)
ax.set_xticks(x, [f"{d[5:]}\n({DOW[pd.Timestamp(d).dayofweek]})" for d in c.index])
ax.set_ylim(0, c.sum(axis=1).max() * 1.18); ax.set_ylabel("행 수")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
ax.set_title("실습 데이터 날짜별 행 수", loc="left")
ax.legend(frameon=False, ncol=2, loc="upper right", bbox_to_anchor=(1, 1.12))
save(fig, "07_실습_날짜별_행수.png")
S["p_dates"] = c.sum(axis=1).astype(int).to_dict()

# 08 tt_sec 분포 실습 vs 대회
fig, ax = plt.subplots(figsize=(8, 3.4)); bins = np.arange(0, 640, 8)
for d, nm, col in [(pall, "실습 (train+test)", CAT[0]), (ftr, "대회 train", CAT[1])]:
    ax.hist(d["tt_sec"].clip(upper=632), bins=bins, density=True, histtype="step", lw=2, color=col, label=nm)
ax.set_xlabel("tt_sec (초, 632초 이상은 끝에 모음)"); ax.set_ylabel("비율 (밀도)")
ax.set_title("tt_sec 분포 비교: 실습 vs 대회", loc="left"); ax.legend(frameon=False)
save(fig, "08_tt분포_실습_대회.png")
S["p_tt"] = pall["tt_sec"].describe().round(1).to_dict()

# 09 실습: 평일/토/일 시간대별 (실습 test는 정답이 있어 일요일을 볼 수 있음)
pall["dtype"] = pall["dow"].map(lambda d: "토" if d == "토" else ("일" if d == "일" else "평일(월~금)"))
g = pall[pall.arrive_hour >= 5].groupby(["dtype", "arrive_hour"])["tt_sec"].mean().unstack(0)
fig, ax = plt.subplots(figsize=(8, 3.6))
for i, d in enumerate(["평일(월~금)", "토", "일"]):
    ax.plot(g.index, g[d], color=CAT[i], lw=2, marker="o", ms=3.5, label=d)
ax.set_xticks(range(5, 24, 2)); ax.set_xlabel("출발 정류장 도착 시각 (시)"); ax.set_ylabel("평균 tt_sec (초)")
ax.set_title("실습 데이터 시간대별 평균 통행시간 — 평일·토·일", loc="left")
ax.set_ylim(70, 142); ax.legend(frameon=False, ncol=3, loc="upper right", bbox_to_anchor=(1, 1.13))
save(fig, "09_실습_시간대_평일토일.png")
S["p_hour"] = g.round(1).to_dict()
S["p_daymean"] = pall.groupby("dtype")["tt_sec"].mean().round(1).to_dict()

# 10 실습 노선별
r = pall.groupby("route_nm").agg(n=("tt_sec", "size"), tt=("tt_sec", "mean"), dist=("dist_m", "mean"))
r["sec100"] = pall.groupby("route_nm").apply(lambda d: d.tt_sec.sum() / d.dist_m.sum() * 100)
r = r.sort_values("tt")
fig, axes = plt.subplots(1, 2, figsize=(9, 3.0))
for ax, col, t, fmt in [(axes[0], "tt", "실습 노선별 평균 tt_sec (초)", "{:.0f}"),
                        (axes[1], "sec100", "실습 노선별 100m당 소요시간 (초)", "{:.1f}")]:
    ax.barh(r.index, r[col], color=CAT[0], height=0.6)
    for i, v in enumerate(r[col]): ax.text(v, i, " " + fmt.format(v), va="center", fontsize=8, color=INK2)
    ax.set_title(t, loc="left", fontsize=10); ax.grid(axis="y", visible=False); ax.set_xlim(0, r[col].max() * 1.18)
axes[0].set_ylabel("노선"); save(fig, "10_실습_노선별.png")
S["p_route"] = r.round(1).reset_index().to_dict("records")

# 11 대회 train vs test 노선별 행 수
a = pd.DataFrame({"train": ftr.route_nm.value_counts(), "test": fte.route_nm.value_counts()}).sort_values("train")
fig, ax = plt.subplots(figsize=(8, 3.2)); y = np.arange(len(a)); h = 0.38
ax.barh(y + h / 2, a["train"], height=h - 0.03, color=CAT[0], label="train (4일)")
ax.barh(y - h / 2, a["test"], height=h - 0.03, color=CAT[1], label="test (3일)")
for i, (t1, t2) in enumerate(zip(a["train"], a["test"])):
    ax.text(t1, i + h / 2, f" {t1:,}", va="center", fontsize=8, color=INK2)
    ax.text(t2, i - h / 2, f" {t2:,}", va="center", fontsize=8, color=INK2)
ax.set_yticks(y, a.index); ax.set_ylabel("노선"); ax.grid(axis="y", visible=False)
ax.set_xlim(0, a.values.max() * 1.15); ax.set_xlabel("행 수")
ax.set_title("대회 데이터 노선별 행 수 (train vs test)", loc="left"); ax.legend(frameon=False, loc="lower right")
save(fig, "11_대회_노선별_행수.png")
S["f_route_counts"] = a.astype(int).to_dict()

# 12 대회 시간대별 운행량 (하루 평균 행 수)
fall = pd.concat([ftr, fte])
fall["dtype"] = fall["dow"].map(lambda d: "토" if d == "토" else ("일" if d == "일" else "평일(월~금)"))
ndays = fall.groupby("dtype")["date"].nunique()
h = fall.groupby(["dtype", "arrive_hour"]).size().unstack(0).div(ndays).fillna(0)
h = h[h.index >= 5]
fig, ax = plt.subplots(figsize=(8, 3.4))
for i, d in enumerate(["평일(월~금)", "토", "일"]):
    ax.plot(h.index, h[d], color=CAT[i], lw=2, marker="o", ms=3.5, label=d)
ax.set_xticks(range(5, 24, 2)); ax.set_xlabel("시각 (시)"); ax.set_ylabel("하루 평균 행 수")
ax.set_title("대회 데이터 시간대별 운행 기록 수 (train+test)", loc="left"); ax.legend(frameon=False, ncol=3)
save(fig, "12_대회_시간대별_운행량.png")

# 13 실습 실험 흐름 (예상 MAE)
res = pd.read_csv(f"{EXP}/results.csv")
p = res[(res.phase == "practice") & res["예상"].notna()].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(8, 8.2)); y = np.arange(len(p))[::-1]
best = p["예상"].cummin()
ax.plot(best.values, y, color=GRAY, lw=1.5, drawstyle="steps-post", label="그때까지 최고")
ax.scatter(p["예상"], y, s=34, color=CAT[0], zorder=3, edgecolor="#fcfcfb", linewidth=1, label="실험 결과")
ib = p["예상"].idxmin()
ax.scatter([p.loc[ib, "예상"]], [y[ib]], s=60, color=CAT[1], zorder=4, edgecolor="#fcfcfb", label=f"최고 {p.loc[ib, '실험'].split()[0]}")
ax.set_yticks(y, p["실험"], fontsize=8); ax.grid(axis="y", visible=False)
ax.set_xlabel("검증 예상 MAE (낮을수록 좋음)  = 평일×0.72 + 토×0.28")
ax.set_xlim(28.5, 29.5); ax.text(29.48, y[0], f"E0 {p.loc[0,'예상']:.2f} →", ha="right", va="center", fontsize=8, color=INK2)
ax.set_title("실습 단계 실험별 검증 MAE (E0 → E33)", loc="left"); ax.legend(frameon=False, loc="lower right")
save(fig, "13_실습_실험흐름.png")
S["p_exp_best"] = p.loc[ib, ["실험", "예상"]].to_dict()

# 14 대회 단계 실험
f = res[(res.phase == "final") & res["예상"].notna() & (res["예상"] < 30)].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(8, 7.2)); y = np.arange(len(f))[::-1]
base = 25.421
cols = [CAT[0] if v < base else GRAY for v in f["예상"]]
ax.barh(y, f["예상"] - base, left=base, color=cols, height=0.65)
for yi, v in zip(y, f["예상"]):
    ax.text(v + (0.008 if v >= base else -0.008), yi, f"{v:.3f}", va="center",
            ha="left" if v >= base else "right", fontsize=7.5, color=INK2)
ax.axvline(base, color="#0b0b0b", lw=1)
ax.set_yticks(y, f["실험"], fontsize=8); ax.grid(axis="y", visible=False)
ax.set_xlim(25.1, 25.95); ax.set_xlabel("검증 예상 MAE (세로선 = 기준 E38A 25.421)")
ax.legend(handles=[Patch(color=CAT[0], label="기준보다 좋음"), Patch(color=GRAY, label="기준보다 나쁨")],
          frameon=False, loc="lower right")
ax.set_title("대회 단계 실험별 검증 MAE (E38 → E50)", loc="left")
save(fig, "14_대회_실험별.png")
S["f_exp"] = f[["실험", "평일", "예상"]].to_dict("records")

# 15 public 제출 기록
pub = pd.DataFrame([("E32 기본", 25.028), ("E35 일요일 섞기", 24.808), ("E37 적응형 (제외)", 24.798),
                    ("E41 회차 힌트", 24.610), ("E46 평일 전용+노선별", 24.577)], columns=["제출", "mae"])
fig, ax = plt.subplots(figsize=(8, 3.4)); x = np.arange(len(pub))
ax.plot(x, pub["mae"], color=CAT[0], lw=2, marker="o", ms=7)
for i, v in enumerate(pub["mae"]):
    d = "" if i == 0 else f"\nE32 대비 {v - pub['mae'][0]:+.3f}"
    ax.annotate(f"{v:.3f}{d}", (i, v), textcoords="offset points", xytext=(0, 14), ha="center", fontsize=8, color=INK2)
ax.set_xticks(x, pub["제출"]); ax.set_ylim(24.45, 25.2); ax.set_ylabel("public MAE")
ax.set_title("대회 제출 public MAE 변화 (낮을수록 좋음)", loc="left")
save(fig, "15_public_제출기록.png")

# 16 E47 비교
e47 = pd.read_csv(f"{EXP}/E47_cv_table.csv").iloc[1:]
e47 = e47.sort_values("E41 최종 대비", ascending=False)
fig, ax = plt.subplots(figsize=(8, 3.4)); y = np.arange(len(e47))
v = e47["E41 최종 대비"]
ax.barh(y, v, color=[CAT[0] if t < 0 else GRAY for t in v], height=0.6)
for yi, t in zip(y, v):
    ax.text(t + (0.01 if t >= 0 else -0.01), yi, f"{t:+.3f}", va="center", ha="left" if t >= 0 else "right", fontsize=8, color=INK2)
ax.axvline(0, color="#0b0b0b", lw=1); ax.set_yticks(y, e47["실험"], fontsize=8.5); ax.grid(axis="y", visible=False)
ax.set_xlim(-0.12, 0.72); ax.set_xlabel("E41 최종 대비 검증 예상 MAE 차이 (음수 = 좋아짐)")
ax.set_title("E47 실험: E41 최종 모델과 비교", loc="left")
save(fig, "16_E47_비교.png")

# 17 회차 규칙: 모델 비율별 (CV vs 실습 test)
cv = pd.read_csv(f"{EXP}/turn_rule_cv.csv"); ts = pd.read_csv(f"{EXP}/turn_rule_test.csv")
def fam(df, col):
    rows = []
    for m, val in zip(df["방법"], df[col]):
        if "그대로" in m: continue
        k = m.split(" 섞기")[0]; w = float(m.split("모델 ")[1].rstrip(")")) if "섞기" in m else 0.0
        rows.append((k, w, val))
    o = pd.DataFrame(rows, columns=["k", "w", "v"])
    base = df.loc[df["방법"].str.contains("그대로"), col].iloc[0]
    return o, base
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharex=True)
fams = ["A 노선×시간", "A3 노선×시간(±1h)", "B 노선×요일(평/토/일)×시간", "B2 노선×평일/주말×시간", "C A3×차량 보정"]
for ax, (df, col, t) in zip(axes, [(cv, "예상", "train 검증 (예상)"), (ts, "test", "실습 test (정답)")]):
    o, base = fam(df, col)
    for i, k in enumerate(fams):
        s = o[o.k == k].sort_values("w")
        s = pd.concat([s, pd.DataFrame([(k, 1.0, base)], columns=s.columns)])
        ax.plot(s.w, s.v, color=CAT[i], lw=2, marker="o", ms=4, label=k.split()[0])
    ax.set_title(t, loc="left", fontsize=10); ax.set_xlabel("모델 예측 비율 (1 = 모델만, 0 = 규칙만)")
axes[0].set_ylabel("MAE"); axes[1].legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
fig.suptitle("회차 구간 규칙 섞기 실험 (실습, E33)", x=0.01, ha="left", fontsize=11)
save(fig, "17_회차규칙_섞기.png")

# 18 E37 토요일 점검 + E51 날짜별
sat = pd.read_csv(f"{EXP}/E37_saturday_check.csv")
e51 = pd.read_csv(f"{EXP}/E51_cv_table.csv").iloc[:3]
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
ax = axes[0]; y = np.arange(len(sat))[::-1]
ax.barh(y, sat["delta_vs_E32"], color=[CAT[0] if t < 0 else GRAY for t in sat["delta_vs_E32"]], height=0.6)
for yi, t in zip(y, sat["delta_vs_E32"]): ax.text(t - 0.0005, yi, f"{t:+.4f}", va="center", ha="right", fontsize=8, color=INK2)
ax.set_yticks(y, ["E32 그대로", "고정 50:50", "고정 70:30", "적응형"]); ax.grid(axis="y", visible=False)
ax.set_xlim(-0.026, 0.002); ax.axvline(0, color="#0b0b0b", lw=1)
ax.set_title("E37 토요일 점검: E32 대비 MAE 차이", loc="left", fontsize=10)
ax = axes[1]; x = np.arange(3); w = 0.38
for j, (c_, nm) in enumerate([("일반 구간 개선폭", "일반 구간"), ("회차 구간 개선폭", "회차 구간")]):
    ax.bar(x + (j - 0.5) * w, e51[c_], width=w - 0.04, color=CAT[j], label=nm)
ax.axhline(0, color="#0b0b0b", lw=1); ax.set_xticks(x, e51["날짜"])
ax.set_ylabel("개선폭 (초, +면 좋아짐)"); ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title("E51 직전 시간대 피처: 날짜별 개선폭", loc="left", fontsize=10)
save(fig, "18_E37_E51_점검.png")

# 19 최종 제출 예측 (E47 best) 시간대별
sub = pd.read_csv(f"{SUB}/final_E47_best.csv").merge(fte[["row_id", "dow", "arrive_hour"]], on="row_id")
q = sub[sub.arrive_hour >= 5].groupby(["dow", "arrive_hour"])["tt_sec"].mean().unstack(0)
tw = ftr[(ftr.arrive_hour >= 5) & (ftr.dow != "토")].groupby("arrive_hour")["tt_sec"].mean()
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(tw.index, tw.values, color=GRAY, lw=1.5, ls="--", label="train 평일 실제")
for i, d in enumerate(["목", "금", "일"]):
    ax.plot(q.index, q[d], color=CAT[i], lw=2, marker="o", ms=3.5, label=f"{d} 예측")
ax.set_xticks(range(5, 24, 2)); ax.set_xlabel("시각 (시)"); ax.set_ylabel("평균 tt_sec (초)")
ax.set_title("최종 제출(E47 best) 예측값 — 시간대별 평균", loc="left"); ax.legend(frameon=False, ncol=4, fontsize=8.5)
save(fig, "19_E47_예측_시간대별.png")
S["sub_mean"] = sub.groupby("dow")["tt_sec"].mean().round(1).to_dict()

# 20 제출 파일끼리 비교: E32 기본 대비 얼마나 바뀌었나 (요일별 평균 절대 차이)
files = ["final_E35_sunmix", "final_E37_adaptive_sunmix", "final_E41_turnhint_sunmix",
         "final_E44_weekday_lgb", "final_E46_wd_route_sunmix", "final_E47_best"]
b = pd.read_csv(f"{SUB}/final_E32.csv").merge(fte[["row_id", "dow"]], on="row_id")
rows = []
for fn in files:
    d = pd.read_csv(f"{SUB}/{fn}.csv").merge(b, on="row_id", suffixes=("", "_b"))
    d["ad"] = (d["tt_sec"] - d["tt_sec_b"]).abs()
    for dw, v in d.groupby("dow")["ad"].mean().items(): rows.append((fn.replace("final_", ""), dw, v))
m = pd.DataFrame(rows, columns=["f", "d", "v"]).pivot(index="f", columns="d", values="v").loc[[f.replace("final_", "") for f in files]]
fig, ax = plt.subplots(figsize=(8, 3.6)); x = np.arange(len(m)); w = 0.27
for j, d in enumerate(["목", "금", "일"]):
    ax.bar(x + (j - 1) * w, m[d], width=w - 0.03, color=CAT[j], label=d)
ax.set_xticks(x, m.index, rotation=15, ha="right", fontsize=8.5)
ax.set_ylabel("E32 대비 평균 |차이| (초)"); ax.legend(frameon=False, ncol=3, loc="upper left")
ax.set_title("제출 파일별로 E32 기본 예측에서 얼마나 바뀌었나 (요일별)", loc="left")
save(fig, "20_제출파일_비교.png")
S["sub_diff"] = m.round(2).to_dict()

print(json.dumps(S, ensure_ascii=False, indent=1, default=str))
