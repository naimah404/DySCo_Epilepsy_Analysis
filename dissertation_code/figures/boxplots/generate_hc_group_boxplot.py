"""
generate_hc_group_boxplot.py
=============================
Three-condition boxplot (Video / Please Wait / Rest) for all healthy-control
participants (n=19). Mirrors generate_adult_group_boxplot.py exactly.
"""

import os
import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── CONFIG ────────────────────────────────────────────────────────────────────
OUTPUT_ROOT = r"D:\adult_controls\Adult\dysco_results"

HALF_WIN = 10
LAG      = 20

VIDEO_BLOCKS_VOL = [(0, 111), (150, 261)]
WAIT_BLOCKS_VOL  = [(111, 150), (261, 300)]

METRICS = [
    ("entropy", "Von Neumann Entropy"),
    ("norm2",   "Connectivity Norm (L2)"),
    ("speed",   "Reconfiguration Speed"),
]

CONDITIONS = ["video", "wait", "rest"]

NAVY    = "#1a2e4a"
C_VIDEO = "#2166ac"
C_WAIT  = "#e08214"
C_REST  = "#d6604d"

FILL  = {"video": "#d1e5f0", "wait": "#fee0b6", "rest": "#fddbc7"}
EDGE  = {"video": C_VIDEO,   "wait": C_WAIT,    "rest": C_REST}
DOT   = {"video": C_VIDEO,   "wait": C_WAIT,    "rest": "#b2182b"}
XLBLS = {
    "video": "Video\n(cartoon)",
    "wait":  "Please wait\n(within cartoon)",
    "rest":  "Rest",
}

# ── helpers ───────────────────────────────────────────────────────────────────

def _win_condition_en(i):
    c = i + HALF_WIN
    if any(s <= c < e for s, e in VIDEO_BLOCKS_VOL): return "video"
    if any(s <= c < e for s, e in WAIT_BLOCKS_VOL):  return "wait"
    return None

def _win_condition_spd(i):
    c = i + HALF_WIN + LAG
    if any(s <= c < e for s, e in VIDEO_BLOCKS_VOL): return "video"
    if any(s <= c < e for s, e in WAIT_BLOCKS_VOL):  return "wait"
    return None

def _is_cartoon(fname): return bool(re.search(r"task-cartoon", fname))
def _is_rest(fname):    return bool(re.search(r"task-rest",    fname))

# ── load data ─────────────────────────────────────────────────────────────────

pid_dirs = sorted(
    d for d in os.listdir(OUTPUT_ROOT)
    if d.startswith("sub-hc") and d.endswith("_dysco_output")
    and os.path.isdir(os.path.join(OUTPUT_ROOT, d))
)

group_pools = {m: {c: [] for c in CONDITIONS} for m, _ in METRICS}
part_pools  = {}

for dname in pid_dirs:
    pid_short = dname.replace("sub-", "").replace("_dysco_output", "")
    folder    = os.path.join(OUTPUT_ROOT, dname)
    npy_files = glob.glob(os.path.join(folder, "*_dysco.npy"))

    pools = {m: {c: [] for c in CONDITIONS} for m, _ in METRICS}

    for fpath in npy_files:
        fname = os.path.basename(fpath)
        try:
            d = np.load(fpath, allow_pickle=True).item()
        except Exception as e:
            print(f"  Could not load {fname}: {e}"); continue

        if _is_cartoon(fname):
            for metric, _ in METRICS:
                if metric not in d: continue
                arr = np.asarray(d[metric], dtype=float)
                if arr.ndim != 1 or len(arr) == 0: continue
                assign_fn = _win_condition_spd if metric == "speed" else _win_condition_en
                for i, val in enumerate(arr):
                    cond = assign_fn(i)
                    if cond:
                        pools[metric][cond].append(val)

        elif _is_rest(fname):
            for metric, _ in METRICS:
                if metric not in d: continue
                arr = np.asarray(d[metric], dtype=float)
                if arr.ndim == 1 and len(arr) > 0:
                    pools[metric]["rest"].extend(arr.tolist())

    if any(pools[m][c] for m, _ in METRICS for c in CONDITIONS):
        part_pools[pid_short] = pools
        for m, _ in METRICS:
            for c in CONDITIONS:
                group_pools[m][c].extend(pools[m][c])
        print(f"  {pid_short}: {len(npy_files)} run(s) loaded")
    else:
        print(f"  {pid_short}: no usable data — skipped")

if not part_pools:
    print("ERROR: no participant data found."); raise SystemExit(1)

pids   = sorted(part_pools.keys())
n_part = len(pids)
print(f"\n{n_part} participants: {pids}\n")

# ── participant medians ───────────────────────────────────────────────────────

part_medians = {
    m: {
        c: [np.median(part_pools[p][m][c]) for p in pids if part_pools[p][m][c]]
        for c in CONDITIONS
    }
    for m, _ in METRICS
}

n_counts = {m: {c: len(group_pools[m][c]) for c in CONDITIONS} for m, _ in METRICS}

# ── figure ────────────────────────────────────────────────────────────────────

rng = np.random.default_rng(42)
fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.patch.set_facecolor("white")

for ax, (metric, ylabel) in zip(axes, METRICS):
    ax.set_facecolor("white")

    box_data = [np.array(group_pools[metric][c]) for c in CONDITIONS]

    bp = ax.boxplot(
        box_data,
        positions=[1, 2, 3],
        widths=0.52,
        patch_artist=True,
        showfliers=True,
        medianprops=dict(color="white", linewidth=2.4, solid_capstyle="round"),
        whiskerprops=dict(linewidth=1.2, color="#555555"),
        capprops=dict(linewidth=1.2, color="#555555"),
        flierprops=dict(marker="o", markersize=2.2, alpha=0.18,
                        markeredgewidth=0, linestyle="none"),
        zorder=2,
    )

    for box, flier, cond in zip(bp["boxes"], bp["fliers"], CONDITIONS):
        box.set_facecolor(FILL[cond])
        box.set_edgecolor(EDGE[cond])
        box.set_linewidth(1.7)
        box.set_zorder(3)
        flier.set_markerfacecolor(EDGE[cond])
        flier.set_markeredgecolor("none")

    for pos, cond in zip([1, 2, 3], CONDITIONS):
        meds   = part_medians[metric][cond]
        jitter = rng.uniform(-0.06, 0.06, size=len(meds))
        ax.scatter([pos + j for j in jitter], meds,
                   color=DOT[cond], s=48, zorder=6,
                   edgecolors="white", linewidths=0.9)

    tick_labels = [
        f"{XLBLS[c]}\n$n$ = {n_counts[metric][c]:,}"
        for c in CONDITIONS
    ]
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(tick_labels, fontsize=9.5, color=NAVY, linespacing=1.5)
    ax.set_xlim(0.4, 3.6)
    ax.set_ylabel(ylabel, fontsize=10.5, color=NAVY, labelpad=6)
    ax.set_title(ylabel, fontsize=12, fontweight="bold", color=NAVY, pad=10)
    ax.tick_params(axis="y", labelcolor=NAVY, labelsize=9)
    ax.tick_params(axis="x", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#bbbbbb")
    ax.spines["bottom"].set_color("#bbbbbb")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#dddddd", zorder=0)
    ax.set_axisbelow(True)

fig.suptitle(
    "DySCo Metrics Across Conditions: Video, Please Wait, and Rest — Healthy Controls",
    fontsize=14, fontweight="bold", color=NAVY, y=1.02
)
fig.text(
    0.5, -0.06,
    f"Boxes show group-level window distributions (IQR ± 1.5×IQR).  "
    f"Dots show individual participant medians (n = {n_part}).  "
    "Windows assigned by sliding-window centre volume.",
    ha="center", fontsize=8.5, color="#555555", style="italic"
)

plt.tight_layout()

save_path = os.path.join(OUTPUT_ROOT, "hc_group_boxplot_by_condition.png")
fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {save_path}")
