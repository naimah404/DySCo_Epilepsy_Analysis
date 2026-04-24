"""
generate_cross_group_comparison.py
====================================
Side-by-side boxplot comparing paediatric epilepsy (n=5) vs adult epilepsy
(n=10) for each DySCo metric (entropy, speed, norm2) across three conditions
(Video, Please Wait, Rest).

Layout: 3 metric panels, 3 condition clusters per panel, 2 boxes per cluster
(paediatric | adult), individual participant median dots overlaid.

Saved to:
  DELFT_NEW/dissertation_figures_final/cross_group_comparison.png
"""

import os
import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── CONFIG ────────────────────────────────────────────────────────────────────
PAED_ROOT  = r"C:\Users\naima\DySCo-main\DySCo-main\DELFT_NEW"
ADULT_ROOT = r"D:\encrypt_generalised_adult\ADULT\dysco_results"
OUT_DIR    = r"C:\Users\naima\DySCo-main\DySCo-main\DELFT_NEW\dissertation_figures_final"
os.makedirs(OUT_DIR, exist_ok=True)

HALF_WIN = 10
LAG      = 20

VIDEO_BLOCKS_VOL = [(0, 111), (150, 261)]
WAIT_BLOCKS_VOL  = [(111, 150), (261, 300)]   # adult goes to 300

METRICS = [
    ("entropy", "Von Neumann Entropy"),
    ("speed",   "Reconfiguration Speed"),
    ("norm2",   "Connectivity Norm (L2)"),
]
CONDITIONS = ["video", "wait", "rest"]

NAVY = "#1a2e4a"

# Paediatric: blue family  |  Adult: orange/red family
PAED_FILL  = {"video": "#c6dbef", "wait": "#fdd0a2", "rest": "#fcbba1"}
PAED_EDGE  = {"video": "#2166ac", "wait": "#d6604d", "rest": "#b2182b"}
ADULT_FILL = {"video": "#6baed6", "wait": "#fd8d3c", "rest": "#ef3b2c"}
ADULT_EDGE = {"video": "#084594", "wait": "#a63603", "rest": "#67000d"}
PAED_DOT   = {"video": "#2166ac", "wait": "#d6604d", "rest": "#b2182b"}
ADULT_DOT  = {"video": "#084594", "wait": "#a63603", "rest": "#67000d"}

# ── window condition helpers ──────────────────────────────────────────────────

def _cond_en(i):
    c = i + HALF_WIN
    if any(s <= c < e for s, e in VIDEO_BLOCKS_VOL): return "video"
    if any(s <= c < e for s, e in WAIT_BLOCKS_VOL):  return "wait"
    return None

def _cond_spd(i):
    c = i + HALF_WIN + LAG
    if any(s <= c < e for s, e in VIDEO_BLOCKS_VOL): return "video"
    if any(s <= c < e for s, e in WAIT_BLOCKS_VOL):  return "wait"
    return None

# ── data loader ───────────────────────────────────────────────────────────────

def _load_group(npy_files_by_pid, is_cartoon_fn):
    """
    Returns {pid: {metric: {condition: [values]}}}
    is_cartoon_fn(fname) → True/False
    """
    group = {}
    for pid, files in npy_files_by_pid.items():
        pools = {m: {c: [] for c in CONDITIONS} for m, _ in METRICS}
        for fpath in files:
            fname = os.path.basename(fpath)
            try:
                d = np.load(fpath, allow_pickle=True).item()
            except Exception as e:
                print(f"  Could not load {fname}: {e}")
                continue

            if is_cartoon_fn(fname):
                for metric, _ in METRICS:
                    if metric not in d:
                        continue
                    arr  = np.asarray(d[metric], dtype=float)
                    afn  = _cond_spd if metric == "speed" else _cond_en
                    for i, val in enumerate(arr):
                        cond = afn(i)
                        if cond:
                            pools[metric][cond].append(val)
            else:   # rest run
                for metric, _ in METRICS:
                    if metric not in d:
                        continue
                    arr = np.asarray(d[metric], dtype=float)
                    pools[metric]["rest"].extend(arr.tolist())

        group[pid] = pools
    return group


# ── paediatric: discover files ────────────────────────────────────────────────
paed_files = {}
for pid_dir in sorted(glob.glob(os.path.join(PAED_ROOT, "P*_dysco_output"))):
    pid = os.path.basename(pid_dir).replace("_dysco_output", "")
    npy = glob.glob(os.path.join(pid_dir, "*_dysco.npy"))
    if npy:
        paed_files[pid] = npy

def _is_paed_cartoon(fname):
    return bool(re.search(r"_c\d+_", fname))

paed_data = _load_group(paed_files, _is_paed_cartoon)
print(f"Paediatric: {sorted(paed_data.keys())}")


# ── adult: discover files ─────────────────────────────────────────────────────
adult_files = {}
for pid_dir in sorted(glob.glob(os.path.join(ADULT_ROOT, "sub-ga*_dysco_output"))):
    pid_short = os.path.basename(pid_dir).replace("sub-", "").replace("_dysco_output", "")
    # ses-01 only for consistency
    npy = [f for f in glob.glob(os.path.join(pid_dir, "*_dysco.npy"))
           if "ses-01" in os.path.basename(f)]
    if npy:
        adult_files[pid_short] = npy

def _is_adult_cartoon(fname):
    return "task-cartoon" in fname

adult_data = _load_group(adult_files, _is_adult_cartoon)
print(f"Adult: {sorted(adult_data.keys())}")


# ── participant-level medians ─────────────────────────────────────────────────
def _part_medians(group_data, metric, cond):
    return [np.median(group_data[p][metric][cond])
            for p in sorted(group_data)
            if group_data[p][metric][cond]]

def _group_pool(group_data, metric, cond):
    vals = []
    for p in group_data:
        vals.extend(group_data[p][metric][cond])
    return np.array(vals)


# ── figure ────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

COND_LABELS = {"video": "Video\n(cartoon)", "wait": "Please Wait\n(within cartoon)", "rest": "Rest"}

fig, axes = plt.subplots(1, 3, figsize=(17, 7))
fig.patch.set_facecolor("white")

for ax, (metric, ylabel) in zip(axes, METRICS):
    ax.set_facecolor("white")

    xtick_pos, xtick_lbl = [], []
    cluster_gap = 3.0   # space between condition clusters
    box_width   = 0.7
    box_gap     = 0.9   # gap between paed and adult box within a cluster

    for ci, cond in enumerate(CONDITIONS):
        base = ci * cluster_gap
        pos_p = base             # paediatric position
        pos_a = base + box_gap   # adult position

        for pos, fill, edge, dot, gdata, label in [
            (pos_p, PAED_FILL[cond],  PAED_EDGE[cond],  PAED_DOT[cond],  paed_data,  "Paediatric"),
            (pos_a, ADULT_FILL[cond], ADULT_EDGE[cond], ADULT_DOT[cond], adult_data, "Adult"),
        ]:
            pool = _group_pool(gdata, metric, cond)
            if len(pool) == 0:
                continue
            bp = ax.boxplot(
                [pool], positions=[pos], widths=box_width,
                patch_artist=True, showfliers=True,
                medianprops=dict(color="white", linewidth=2.2, solid_capstyle="round"),
                whiskerprops=dict(linewidth=1.1, color="#555555"),
                capprops=dict(linewidth=1.1, color="#555555"),
                flierprops=dict(marker="o", markersize=2, alpha=0.15,
                                markeredgewidth=0, linestyle="none"),
                zorder=2,
            )
            bp["boxes"][0].set_facecolor(fill)
            bp["boxes"][0].set_edgecolor(edge)
            bp["boxes"][0].set_linewidth(1.6)
            bp["boxes"][0].set_zorder(3)
            bp["fliers"][0].set_markerfacecolor(edge)
            bp["fliers"][0].set_markeredgecolor("none")

            meds   = _part_medians(gdata, metric, cond)
            jitter = rng.uniform(-0.1, 0.1, size=len(meds))
            ax.scatter([pos + j for j in jitter], meds,
                       color=dot, s=45, zorder=6,
                       edgecolors="white", linewidths=0.8)

        # x-tick at cluster centre
        xtick_pos.append(base + box_gap / 2)
        n_p = len(_part_medians(paed_data, metric, cond))
        n_a = len(_part_medians(adult_data, metric, cond))
        xtick_lbl.append(f"{COND_LABELS[cond]}\nP n={n_p}  A n={n_a}")

    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_lbl, fontsize=8.5, color=NAVY, linespacing=1.4)
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

# legend
legend_handles = [
    mpatches.Patch(facecolor="#aec8e8", edgecolor="#2166ac", linewidth=1.5,
                   label=f"Paediatric epilepsy (n = {len(paed_data)})"),
    mpatches.Patch(facecolor="#6baed6", edgecolor="#084594", linewidth=1.5,
                   label=f"Adult epilepsy (n = {len(adult_data)})"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=10,
           frameon=True, framealpha=0.95, edgecolor="#cccccc",
           bbox_to_anchor=(0.5, -0.04))

fig.suptitle(
    "DySCo Metrics Across Conditions: Paediatric vs Adult Generalised Epilepsy",
    fontsize=14, fontweight="bold", color=NAVY, y=1.02
)
fig.text(
    0.5, -0.10,
    "Boxes = group-level window distributions (IQR ± 1.5×IQR).  "
    "Dots = individual participant medians.  "
    "Video and Wait windows assigned by Shamshiri et al. (2016) block boundaries.",
    ha="center", fontsize=8, color="#555555", style="italic"
)

plt.tight_layout()
save_path = os.path.join(OUT_DIR, "cross_group_comparison.png")
fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"\nSaved: {save_path}")
