"""
Three-condition boxplot: Video vs Please Wait vs Rest
Window-level data pooled across all participants for the box distributions.
Participant-level medians (n=5 per condition) overlaid as dots.

Shamshiri et al. block structure (volumes):
  Video 1:   0–111
  Wait 1:    111–150
  Video 2:   150–261
  Wait 2:    261–296

Window assignment uses the centre volume of each sliding window.
Half-window = 10, so window i has centre at volume i + 10.
Speed[i] is assigned using the centre of the later window: (i + 20) + 10.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import os

# ── paths ─────────────────────────────────────────────────────────────────────
BASE = r"C:\Users\naima\DySCo-main\DySCo-main\DELFT_NEW"
OUT  = r"C:\Users\naima\DySCo-main\DySCo-main\DELFT_NEW\dissertation_figures_final"
os.makedirs(OUT, exist_ok=True)

# ── participant run catalogue ─────────────────────────────────────────────────
RUNS = {
    "P001": {"cartoon": ["c003", "c005"], "rest": ["r004", "r006"]},
    "P002": {"cartoon": ["c003", "c005"], "rest": ["r004", "r006"]},
    "P003": {"cartoon": ["c003", "c006"], "rest": ["r004", "r007"]},
    "P004": {"cartoon": ["c004", "c005"], "rest": ["r003"]},
   
}
PARTICIPANTS = list(RUNS.keys())

# ── Shamshiri block boundaries ────────────────────────────────────────────────
VIDEO_BLOCKS = [(0, 111), (150, 261)]
WAIT_BLOCKS  = [(111, 150), (261, 296)]
HALF_WIN     = 10
LAG          = 20

def condition_label_en(i):
    centre = i + HALF_WIN
    for s, e in VIDEO_BLOCKS:
        if s <= centre < e:
            return "video"
    for s, e in WAIT_BLOCKS:
        if s <= centre < e:
            return "wait"
    return None

def condition_label_spd(i):
    centre = i + LAG + HALF_WIN
    for s, e in VIDEO_BLOCKS:
        if s <= centre < e:
            return "video"
    for s, e in WAIT_BLOCKS:
        if s <= centre < e:
            return "wait"
    return None

# ── accumulate: group pools + per-participant pools ───────────────────────────
CONDITIONS = ["video", "wait", "rest"]
METRICS    = ["entropy", "norm2", "speed"]

# group-level (all windows)
pools = {m: {c: [] for c in CONDITIONS} for m in METRICS}

# per-participant (windows per participant per condition → median)
part_pools = {
    pid: {m: {c: [] for c in CONDITIONS} for m in METRICS}
    for pid in PARTICIPANTS
}

for pid, run_info in RUNS.items():
    pnum   = pid.lower()
    folder = os.path.join(BASE, f"{pid}_dysco_output")

    for run_id in run_info["cartoon"]:
        fpath = os.path.join(folder, f"{pnum}_{run_id}_merged_dysco.npy")
        if not os.path.exists(fpath):
            print(f"WARNING: missing {fpath}")
            continue
        data = np.load(fpath, allow_pickle=True).item()
        entropy = np.array(data["entropy"])
        norm2   = np.array(data["norm2"])
        speed   = np.array(data["speed"])

        for i in range(len(entropy)):
            lbl = condition_label_en(i)
            if lbl:
                pools["entropy"][lbl].append(entropy[i])
                pools["norm2"][lbl].append(norm2[i])
                part_pools[pid]["entropy"][lbl].append(entropy[i])
                part_pools[pid]["norm2"][lbl].append(norm2[i])

        for i in range(len(speed)):
            lbl = condition_label_spd(i)
            if lbl:
                pools["speed"][lbl].append(speed[i])
                part_pools[pid]["speed"][lbl].append(speed[i])

    for run_id in run_info["rest"]:
        fpath = os.path.join(folder, f"{pnum}_{run_id}_merged_dysco.npy")
        if not os.path.exists(fpath):
            print(f"WARNING: missing {fpath}")
            continue
        data = np.load(fpath, allow_pickle=True).item()
        for m, key in [("entropy","entropy"), ("norm2","norm2"), ("speed","speed")]:
            vals = list(data[key])
            pools[m]["rest"].extend(vals)
            part_pools[pid][m]["rest"].extend(vals)

# participant-level medians (5 per condition per metric)
part_medians = {
    m: {c: [np.median(part_pools[pid][m][c]) for pid in PARTICIPANTS]
        for c in CONDITIONS}
    for m in METRICS
}

# window counts for n= labels
n_counts = {m: {c: len(pools[m][c]) for c in CONDITIONS} for m in METRICS}

# ── style ─────────────────────────────────────────────────────────────────────
NAVY   = "#1a2e4a"

# Video=dark blue, Please Wait=mid coral (lighter, transitional), Rest=dark red
EDGE  = {"video": "#2166ac", "wait": "#b35806", "rest": "#d6604d"}
FILL  = {"video": "#d1e5f0", "wait": "#fee0b6", "rest": "#fddbc7"}
DOT   = {"video": "#2166ac", "wait": "#b35806", "rest": "#b2182b"}

XLABELS = {
    "video": "Video\n(cartoon)",
    "wait":  "Please wait\n(within cartoon)",
    "rest":  "Rest",
}
PANEL_TITLES = {
    "entropy": "Von Neumann Entropy",
    "speed":   "Reconfiguration Speed",
    "norm2":   "Connectivity Norm (L2)",
}
YLABELS = {
    "entropy": "Von Neumann Entropy",
    "speed":   "Reconfiguration Speed",
    "norm2":   "Connectivity Norm (L2)",
}

rng = np.random.default_rng(42)

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 6))
fig.patch.set_facecolor("white")

POSITIONS = [1, 2, 3]
BOX_WIDTH = 0.45

for ax, metric in zip(axes, METRICS):
    data_by_cond = [np.array(pools[metric][c]) for c in CONDITIONS]

    bp = ax.boxplot(
        data_by_cond,
        positions=POSITIONS,
        widths=BOX_WIDTH,
        patch_artist=True,
        showfliers=True,
        medianprops=dict(color="white", linewidth=2.2, solid_capstyle="round"),
        whiskerprops=dict(linewidth=1.2, color="#555555", linestyle="-"),
        capprops=dict(linewidth=1.2, color="#555555"),
        flierprops=dict(
            marker="o", markersize=2.5, alpha=0.25,
            markeredgewidth=0, linestyle="none"
        ),
        zorder=2,
    )

    for patch, cond in zip(bp["boxes"], CONDITIONS):
        patch.set_facecolor(FILL[cond])
        patch.set_edgecolor(EDGE[cond])
        patch.set_linewidth(1.6)
        patch.set_zorder(3)

    for flier, cond in zip(bp["fliers"], CONDITIONS):
        flier.set_markerfacecolor(EDGE[cond])
        flier.set_markeredgecolor("none")

    # ── participant median dots ───────────────────────────────────────────────
    for pos, cond in zip(POSITIONS, CONDITIONS):
        medians = part_medians[metric][cond]
        jitter  = rng.uniform(-0.07, 0.07, size=len(medians))
        ax.scatter(
            [pos + j for j in jitter], medians,
            color=DOT[cond], s=40, zorder=6,
            edgecolors="white", linewidths=0.8,
        )

    # ── n= labels below x-axis ticks ─────────────────────────────────────────
    tick_labels = [
        f"{XLABELS[c]}\n$n$ = {n_counts[metric][c]:,}"
        for c in CONDITIONS
    ]

    ax.set_xticks(POSITIONS)
    ax.set_xticklabels(tick_labels, fontsize=9, color=NAVY, linespacing=1.5)
    ax.set_xlim(0.4, 3.6)

    ax.set_ylabel(YLABELS[metric], fontsize=10.5, color=NAVY, labelpad=6)
    ax.set_title(PANEL_TITLES[metric], fontsize=12, fontweight="bold",
                 color=NAVY, pad=10)

    ax.tick_params(axis="y", labelcolor=NAVY, labelsize=9)
    ax.tick_params(axis="x", length=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#bbbbbb")
    ax.spines["bottom"].set_color("#bbbbbb")
    ax.set_facecolor("white")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5,
                  color="#dddddd", zorder=0)
    ax.set_axisbelow(True)

# ── title and footnote ────────────────────────────────────────────────────────
fig.suptitle(
    "DySCo Metrics Across Conditions: Video, Please Wait, and Rest",
    fontsize=13, fontweight="bold", color=NAVY, y=1.02
)
fig.text(
    0.5, -0.04,
    "Boxes show group-level window distributions (IQR ± 1.5×IQR). "
    "Dots show individual participant medians (n = 5). "
    "Whisker windows assigned by sliding-window centre volume.",
    ha="center", va="top", fontsize=8, color="#555555", style="italic"
)

plt.tight_layout(rect=[0, 0, 1, 1])
out_path = os.path.join(OUT, "fig_three_condition_boxplots.png")
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_path}")

# ── sanity check ─────────────────────────────────────────────────────────────
for m in METRICS:
    for c in CONDITIONS:
        print(f"  {m:8s} {c:5s}  n={n_counts[m][c]:5d}  "
              f"group_med={np.median(pools[m][c]):.4g}")
