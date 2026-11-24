import sys, json
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ROOT = sys.argv[1]
OUT = f"{ROOT}/그래프"
import os; os.makedirs(OUT, exist_ok=True)

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
BLUE = CAT[0]

tr = pd.read_parquet(f"{ROOT}/대회_학생용/train.parquet")
te = pd.read_parquet(f"{ROOT}/대회_학생용/test.parquet")
tr["route_nm"] = tr["route_nm"].astype(str); te["route_nm"] = te["route_nm"].astype(str)
DOW = ["월", "화", "수", "목", "금", "토", "일"]
for d in (tr, te):
    d["dow"] = pd.to_datetime(d["date"]).dt.dayofweek.map(lambda i: DOW[i])
stats = {}

# 1. 날짜별 행 수
c_tr = tr.groupby("date").size(); c_te = te.groupby("date").size()
dates = sorted(set(c_tr.index) | set(c_te.index))
fig, ax = plt.subplots(figsize=(8, 3.6))
x = np.arange(len(dates))
lab = [f"{d[5:]}\n({DOW[pd.Timestamp(d).dayofweek]})" for d in dates]
ax.bar(x, [c_tr.get(d, 0) for d in dates], color=CAT[0], width=0.7, label="train")
ax.bar(x, [c_te.get(d, 0) for d in dates], color=CAT[1], width=0.7, label="test")
for i, d in enumerate(dates):
    v = c_tr.get(d, 0) or c_te.get(d, 0)
    ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8, color="#52514e")
ax.set_xticks(x, lab); ax.set_ylabel("행 수"); ax.set_title("날짜별 행 수 (대회 데이터)", loc="left")
ax.set_ylim(0, 47000); ax.legend(frameon=False, ncol=2, loc="upper right", bbox_to_anchor=(1, 1.12))
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
fig.tight_layout(); fig.savefig(f"{OUT}/01_날짜별_행수.png"); plt.close(fig)
stats["date_counts"] = {d: int(c_tr.get(d, 0) or c_te.get(d, 0)) for d in dates}

# 2. tt_sec 분포
y = tr["tt_sec"]; p99 = y.quantile(0.99)
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.hist(y.clip(upper=p99), bins=80, color=BLUE, edgecolor="#fcfcfb", linewidth=0.5)
for q, nm in [(y.median(), "중앙값"), (y.mean(), "평균")]:
    ax.axvline(q, color="#0b0b0b", lw=1, ls="--")
    ax.text(q, ax.get_ylim()[1]*0.95, f" {nm} {q:.0f}초", fontsize=8, va="top")
ax.set_xlabel("tt_sec (초, 상위 1% 잘라냄)"); ax.set_ylabel("행 수")
ax.set_title("구간 통행시간 tt_sec 분포 (train)", loc="left")
fig.tight_layout(); fig.savefig(f"{OUT}/02_tt_sec_분포.png"); plt.close(fig)
stats["tt"] = {k: float(v) for k, v in y.describe(percentiles=[.25, .5, .75, .99]).items()}

# 3. 시간대별 평균 tt_sec, 요일별
g = tr.groupby(["dow", "arrive_hour"])["tt_sec"].mean().unstack(0)
fig, ax = plt.subplots(figsize=(8, 3.8))
for i, d in enumerate(["월", "화", "수", "토"]):
    s = g[d].dropna(); s = s[s.index >= 5]
    ax.plot(s.index, s.values, color=CAT[i], lw=2, marker="o", ms=3.5, label=d)
ax.set_xlabel("출발 정류장 도착 시각 (시)"); ax.set_ylabel("평균 tt_sec (초)")
ax.set_xticks(range(5, 24, 2))
ax.set_title("시간대별 평균 통행시간 — 요일별 (train, 5~23시)", loc="left")
ax.legend(frameon=False, ncol=4, loc="upper right")
fig.tight_layout(); fig.savefig(f"{OUT}/03_시간대별_요일별.png"); plt.close(fig)
stats["hour_dow"] = g.round(1).to_dict()

# 4. 노선별 평균 tt_sec & 100m당 시간
r = tr.groupby("route_nm").agg(n=("tt_sec", "size"), tt=("tt_sec", "mean"),
                               dist=("dist_m", "mean"), segs=("from_stop", "nunique"))
r["sec100"] = tr.groupby("route_nm").apply(lambda d: d["tt_sec"].sum() / d["dist_m"].sum() * 100)
r = r.sort_values("tt")
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
for ax, col, t, fmt in [(axes[0], "tt", "노선별 평균 tt_sec (초)", "{:.0f}"),
                        (axes[1], "sec100", "노선별 100m당 소요시간 (초)", "{:.1f}")]:
    ax.barh(r.index, r[col], color=BLUE, height=0.6)
    for i, v in enumerate(r[col]): ax.text(v, i, " " + fmt.format(v), va="center", fontsize=8, color="#52514e")
    ax.set_title(t, loc="left", fontsize=10); ax.grid(axis="y", visible=False)
    ax.set_xlim(0, r[col].max() * 1.18)
axes[0].set_ylabel("노선")
fig.tight_layout(); fig.savefig(f"{OUT}/04_노선별.png"); plt.close(fig)
stats["route"] = r.round(1).reset_index().to_dict("records")

# 5. 거리 vs 통행시간 (거리 구간별 중앙값 + IQR)
bins = [0, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000, 5000]
tr["dbin"] = pd.cut(tr["dist_m"], bins)
q = tr.groupby("dbin", observed=True)["tt_sec"].quantile([.25, .5, .75]).unstack()
fig, ax = plt.subplots(figsize=(8, 3.6))
xl = [f"{int(b.left)}–{int(b.right)}" for b in q.index]; x = np.arange(len(q))
ax.fill_between(x, q[0.25], q[0.75], color=BLUE, alpha=0.18, lw=0, label="25–75% 범위")
ax.plot(x, q[0.5], color=BLUE, lw=2, marker="o", ms=4, label="중앙값")
ax.set_xticks(x, xl, rotation=30, ha="right"); ax.set_xlabel("정류장 간 거리 dist_m (m)")
ax.set_ylabel("tt_sec (초)"); ax.set_title("거리 구간별 통행시간 (train)", loc="left")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/05_거리별_통행시간.png"); plt.close(fig)
stats["dist"] = {k: v for k, v in zip(xl, q[0.5].round(0).tolist())}
stats["corr"] = float(tr[["dist_m", "tt_sec"]].corr().iloc[0, 1])

# 6. 노선 × 방향 박스플롯
order = r.index.tolist()
fig, ax = plt.subplots(figsize=(8, 3.6))
data, pos, cols = [], [], []
for i, rn in enumerate(order):
    for j, dr in enumerate([0, 1]):
        data.append(tr.loc[(tr.route_nm == rn) & (tr.direction == dr), "tt_sec"].values)
        pos.append(i * 3 + j); cols.append(CAT[j])
bp = ax.boxplot(data, positions=pos, widths=0.8, showfliers=False, patch_artist=True,
                medianprops=dict(color="#0b0b0b", lw=1.2))
for b, c in zip(bp["boxes"], cols): b.set_facecolor(c); b.set_edgecolor(c); b.set_alpha(0.85)
ax.set_xticks([i * 3 + 0.5 for i in range(len(order))], order); ax.set_xlabel("노선")
ax.set_ylabel("tt_sec (초)"); ax.set_title("노선·방향별 tt_sec 분포 (이상치 제외)", loc="left")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=CAT[0], label="상행(0)"), Patch(color=CAT[1], label="하행(1)")],
          frameon=False, ncol=2, loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/06_노선_방향별.png"); plt.close(fig)

print(json.dumps(stats, ensure_ascii=False, indent=1, default=str))
