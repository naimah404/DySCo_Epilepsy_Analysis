"""
Dissertation-quality FCD matrix figures — four conditions:
  1. Overall cartoon runs        (276×276 full matrix, averaged across 10 runs)
  2. Video watching only         (212×212 submatrix — video windows within cartoon)
  3. Please wait only            (64×64 submatrix — waiting period windows)
  4. Rest                        (276×276 full matrix, averaged across 8 runs)

Colourmap: 'turbo' — perceptually-improved full-spectrum rainbow,
           maximises visible contrast across the distance range.

Saved to: dissertation_figures_final/fcd_matrix/
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import os

# ── paths ─────────────────────────────────────────────────────────────────────
BASE = r"C:\Users\naima\DySCo-main\DySCo-main\DELFT_NEW"
OUT  = r"C:\Users\naima\DySCo-main\DySCo-main\DELFT_NEW\dissertation_figures_final\fcd_matrix"
os.makedirs(OUT, exist_ok=True)

RUNS = {
    "P001": {"cartoon": ["c003", "c005"], "rest": ["r004", "r006"]},
    "P002": {"cartoon": ["c003", "c005"], "rest": ["r004", "r006"]},
    "P003": {"cartoon": ["c003", "c006"], "rest": ["r004", "r007"]},
    "P004": {"cartoon": ["c004", "c005"], "rest": ["r003"]},
    "P005": {"cartoon": ["c003", "c006"], "rest": ["r004"]},
}

TR       = 2.16
HALF_WIN = 10
NAVY     = "#1a2e4a"
CMAP = "viridis"

# ── block window indices ──────────────────────────────────────────────────────
VIDEO_BLOCKS_VOL = [(0, 111), (150, 261)]
WAIT_BLOCKS_VOL  = [(111, 150), (261, 296)]
N_WIN = 276

vid_idx, wait_idx = [], []
for i in range(N_WIN):
    c = i + HALF_WIN
    if any(s <= c < e for s, e in VIDEO_BLOCKS_VOL):
        vid_idx.append(i)
    elif any(s <= c < e for s, e in WAIT_BLOCKS_VOL):
        wait_idx.append(i)

vid_idx  = np.array(vid_idx)   # 212 windows
wait_idx = np.array(wait_idx)  # 64 windows

# time in seconds for full axis
t_full = (np.arange(N_WIN) + HALF_WIN) * TR          # 276 values
t_vid  = t_full[vid_idx]                              # 212 values
t_wait = t_full[wait_idx]                             # 64 values

print(f"Video windows: {len(vid_idx)}")
print(f"Wait windows:  {len(wait_idx)}")

# ── load FCD matrices ─────────────────────────────────────────────────────────
fcd_cartoon_all, fcd_rest_all = [], []

for pid, run_info in RUNS.items():
    pnum   = pid.lower()
    folder = os.path.join(BASE, f"{pid}_dysco_output")
    for cond, lst in [("cartoon", fcd_cartoon_all), ("rest", fcd_rest_all)]:
        for run_id in run_info[cond]:
            fpath = os.path.join(folder, f"{pnum}_{run_id}_merged_dysco.npy")
            if not os.path.exists(fpath):
                print(f"  WARNING: missing {fpath}")
                continue
            data = np.load(fpath, allow_pickle=True).item()
            fcd  = np.array(data["fcd"])
            if fcd.shape == (276, 276):
                lst.append(fcd)

print(f"Cartoon runs: {len(fcd_cartoon_all)}")
print(f"Rest runs:    {len(fcd_rest_all)}")

# ── build the four matrices ───────────────────────────────────────────────────
fcd_cartoon_mean = np.mean(fcd_cartoon_all, axis=0)          # 276×276
fcd_rest_mean    = np.mean(fcd_rest_all,    axis=0)          # 276×276

# submatrices: rows AND columns at block-specific indices
fcd_video_sub = fcd_cartoon_mean[np.ix_(vid_idx,  vid_idx)]  # 212×212
fcd_wait_sub  = fcd_cartoon_mean[np.ix_(wait_idx, wait_idx)] # 64×64

matrices = [
    {
        "fcd":    fcd_cartoon_mean,
        "t_row":  t_full,
        "t_col":  t_full,
        "title":  "Overall Cartoon Runs",
        "subtitle": "Full run average  |  10 runs, 5 participants",
        "fname":  "fcd_cartoon_full.png",
        "xlabel": "Time (seconds)",
        "ylabel": "Time (seconds)",
    },
    {
        "fcd":    fcd_video_sub,
        "t_row":  t_vid,
        "t_col":  t_vid,
        "title":  "Video Watching (within Cartoon)",
        "subtitle": "Video block windows only  |  212 windows per run",
        "fname":  "fcd_video_watching.png",
        "xlabel": "Time (seconds) — video windows",
        "ylabel": "Time (seconds) — video windows",
    },
    {
        "fcd":    fcd_wait_sub,
        "t_row":  t_wait,
        "t_col":  t_wait,
        "title":  'Please Wait (within Cartoon)',
        "subtitle": "Waiting period windows only  |  64 windows per run",
        "fname":  "fcd_please_wait.png",
        "xlabel": "Time (seconds) — wait windows",
        "ylabel": "Time (seconds) — wait windows",
    },
    {
        "fcd":    fcd_rest_mean,
        "t_row":  t_full,
        "t_col":  t_full,
        "title":  "Resting State",
        "subtitle": "Full run average  |  8 runs, 5 participants",
        "fname":  "fcd_rest.png",
        "xlabel": "Time (seconds)",
        "ylabel": "Time (seconds)",
    },
]

# shared colour scale across all four — off-diagonal only, 5th/95th percentile
# (diagonal is always 0 so excluding it prevents vmin being pulled to zero)
def _offdiag(mat):
    n = mat.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return mat[mask]

all_offdiag = np.concatenate([_offdiag(m["fcd"]) for m in matrices])
vmin = np.percentile(all_offdiag, 5)
vmax = np.percentile(all_offdiag, 95)

# ── plot each matrix ──────────────────────────────────────────────────────────
def make_tick_labels(t_arr, n_ticks=6):
    idx  = np.linspace(0, len(t_arr) - 1, n_ticks, dtype=int)
    pos  = idx
    labs = [f"{t_arr[i]:.0f}" for i in idx]
    return pos, labs

for m in matrices:
    fcd    = m["fcd"]
    n      = fcd.shape[0]
    t_row  = m["t_row"]
    t_col  = m["t_col"]

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    im = ax.imshow(
        fcd, cmap=CMAP, aspect="auto", origin="lower",
        vmin=vmin, vmax=vmax, interpolation="gaussian"
    )

    # ticks
    x_pos, x_labs = make_tick_labels(t_col)
    y_pos, y_labs = make_tick_labels(t_row)
    ax.set_xticks(x_pos); ax.set_xticklabels(x_labs, fontsize=9.5, color=NAVY)
    ax.set_yticks(y_pos); ax.set_yticklabels(y_labs, fontsize=9.5, color=NAVY)

    ax.set_xlabel(m["xlabel"], fontsize=11, color=NAVY, labelpad=8)
    ax.set_ylabel(m["ylabel"], fontsize=11, color=NAVY, labelpad=8)

    # spines
    for spine in ax.spines.values():
        spine.set_edgecolor("#bbbbbb")
        spine.set_linewidth(0.8)

    ax.tick_params(colors=NAVY, length=4)

    # colourbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, aspect=30)
    cbar.set_label("DySCo Distance", fontsize=10, color=NAVY, labelpad=8)
    cbar.ax.tick_params(labelcolor=NAVY, labelsize=8.5, length=3)
    cbar.outline.set_edgecolor("#bbbbbb")

    # title block
    ax.set_title(
        m["title"],
        fontsize=14, fontweight="bold", color=NAVY, pad=14
    )
    fig.text(
        0.5, 0.97, m["subtitle"],
        ha="center", va="top", fontsize=9, color="#555555", style="italic"
    )
    fig.text(
        0.5, -0.02,
        "Colour encodes DySCo distance between connectivity states  |  "
        "Low (purple) = similar connectivity states  ·  High (yellow) = dissimilar",
        ha="center", fontsize=8, color="#777777", style="italic"
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    out_path = os.path.join(OUT, m["fname"])
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")

# ── bonus: 2×2 overview panel ─────────────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 12))
fig2.patch.set_facecolor("white")
axes2_flat = axes2.flatten()

panel_order = [0, 1, 2, 3]  # cartoon, video, wait, rest
last_im = None

for ax, idx in zip(axes2_flat, panel_order):
    m   = matrices[idx]
    fcd = m["fcd"]
    t_r = m["t_row"]
    t_c = m["t_col"]

    ax.set_facecolor("white")
    last_im = ax.imshow(
        fcd, cmap=CMAP, aspect="auto", origin="lower",
        vmin=vmin, vmax=vmax, interpolation="gaussian"
    )

    x_pos, x_labs = make_tick_labels(t_c, n_ticks=5)
    y_pos, y_labs = make_tick_labels(t_r, n_ticks=5)
    ax.set_xticks(x_pos); ax.set_xticklabels(x_labs, fontsize=9, color=NAVY)
    ax.set_yticks(y_pos); ax.set_yticklabels(y_labs, fontsize=9, color=NAVY)
    ax.set_xlabel("Time (seconds)", fontsize=10, color=NAVY)
    ax.set_ylabel("Time (seconds)", fontsize=10, color=NAVY)
    ax.set_title(m["title"], fontsize=12, fontweight="bold", color=NAVY, pad=10)

    for spine in ax.spines.values():
        spine.set_edgecolor("#bbbbbb")
        spine.set_linewidth(0.8)
    ax.tick_params(colors=NAVY, length=3)

# single shared colourbar on the right
cbar2 = fig2.colorbar(last_im, ax=axes2_flat, fraction=0.015, pad=0.02, aspect=40)
cbar2.set_label("DySCo Distance", fontsize=11, color=NAVY, labelpad=10)
cbar2.ax.tick_params(labelcolor=NAVY, labelsize=9)
cbar2.outline.set_edgecolor("#bbbbbb")

fig2.suptitle(
    "Group Average FCD Matrices — All Conditions  (n = 5 participants)",
    fontsize=15, fontweight="bold", color=NAVY, y=1.01
)
fig2.text(
    0.5, -0.01,
    "Low (blue/purple) = similar connectivity states  ·  High (yellow) = dissimilar  |  "
    "Colour scale matched across all panels",
    ha="center", fontsize=9, color="#777777", style="italic"
)

plt.tight_layout(rect=[0, 0.02, 0.96, 1])
overview_path = os.path.join(OUT, "fcd_all_conditions_overview.png")
fig2.savefig(overview_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig2)
print(f"  Saved: {overview_path}")

print("\nAll FCD figures done.")
