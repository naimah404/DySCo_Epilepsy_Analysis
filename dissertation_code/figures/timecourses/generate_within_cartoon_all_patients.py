"""
Within-cartoon DySCo time-course figure for all 5 patients.
One figure per patient, saved to:
  DELFT_NEW/dissertation_figures_final/within_cartoon_per_patient/

Each figure: 3 panels (entropy, speed, norm2), averaged across that
patient's cartoon runs, with Shamshiri et al. (2016) block overlay.
Line colour switches between video (blue) and waiting period (red).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from scipy.ndimage import uniform_filter1d

# ── paths ─────────────────────────────────────────────────────────────────────
BASE = r"C:\Users\naima\DySCo-main\DySCo-main\DELFT_NEW"
OUT  = r"C:\Users\naima\DySCo-main\DySCo-main\DELFT_NEW\dissertation_figures_final\within_cartoon_per_patient"
os.makedirs(OUT, exist_ok=True)

# ── run catalogue (cartoon runs only) ────────────────────────────────────────
CARTOON_RUNS = {
    "P001": ["c003", "c005"],
    "P002": ["c003", "c005"],
    "P003": ["c003", "c006"],
    "P004": ["c004", "c005"],
    "P005": ["c003", "c006"],
}

TR       = 2.16
HALF_WIN = 10
LAG      = 20
SMOOTH   = 5

# Shamshiri block boundaries (seconds)
VIDEO_BLOCKS = [(0, 240), (324, 564)]
WAIT_BLOCKS  = [(240, 324), (564, 639.4)]

# x-axis end = last window centre (window 275, centre vol 285)
MAX_T = (275 + HALF_WIN) * TR   # 615.6 s

# ── colours ───────────────────────────────────────────────────────────────────
NAVY       = "#1a2e4a"
C_VIDEO_BG = "#d1e5f0"
C_WAIT_BG  = "#fddbc7"
C_VIDEO_LN = "#2166ac"
C_WAIT_LN  = "#d6604d"

# ── per-patient loop ──────────────────────────────────────────────────────────
for pid, run_ids in CARTOON_RUNS.items():
    pnum   = pid.lower()
    folder = os.path.join(BASE, f"{pid}_dysco_output")

    # load all cartoon runs for this patient
    en_list, spd_list, n2_list = [], [], []
    for run_id in run_ids:
        fpath = os.path.join(folder, f"{pnum}_{run_id}_merged_dysco.npy")
        if not os.path.exists(fpath):
            print(f"  WARNING: {fpath} not found, skipping")
            continue
        data = np.load(fpath, allow_pickle=True).item()
        en_list.append(np.array(data["entropy"]))
        spd_list.append(np.array(data["speed"]))
        n2_list.append(np.array(data["norm2"]))

    if not en_list:
        print(f"  No data for {pid}, skipping")
        continue

    n_runs = len(en_list)

    # average across runs (truncate to shortest if lengths differ)
    def avg(lst):
        min_len = min(len(x) for x in lst)
        return np.mean([x[:min_len] for x in lst], axis=0)

    entropy_mean = avg(en_list)
    speed_mean   = avg(spd_list)
    norm2_mean   = avg(n2_list)

    # smooth
    entropy_s = uniform_filter1d(entropy_mean, SMOOTH)
    speed_s   = uniform_filter1d(speed_mean,   SMOOTH)
    norm2_s   = uniform_filter1d(norm2_mean,   SMOOTH)

    # time axes
    t_en  = (np.arange(len(entropy_mean)) + HALF_WIN)       * TR
    t_spd = (np.arange(len(speed_mean))   + HALF_WIN + LAG) * TR

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    fig.patch.set_facecolor("white")

    PANELS = [
        (axes[0], t_en,  entropy_s, "Von Neumann\nEntropy"),
        (axes[1], t_spd, speed_s,   "Reconfiguration\nSpeed"),
        (axes[2], t_en,  norm2_s,   "Connectivity\nNorm (L2)"),
    ]

    for ax, t, signal, ylabel in PANELS:

        # block shading
        for s, e in VIDEO_BLOCKS:
            ax.axvspan(s, min(e, MAX_T), color=C_VIDEO_BG, alpha=0.7, zorder=0)
        for s, e in WAIT_BLOCKS:
            ax.axvspan(s, min(e, MAX_T), color=C_WAIT_BG,  alpha=0.7, zorder=0)

        # line coloured by block
        for i in range(len(t) - 1):
            in_wait = any(s <= t[i] < e for s, e in WAIT_BLOCKS)
            col = C_WAIT_LN if in_wait else C_VIDEO_LN
            ax.plot(t[i:i+2], signal[i:i+2],
                    color=col, linewidth=1.6, solid_capstyle="round", zorder=3)

        ax.set_ylabel(ylabel, fontsize=10, color=NAVY, labelpad=6)
        ax.set_xlim(0, MAX_T)
        ax.tick_params(labelcolor=NAVY, labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#bbbbbb")
        ax.spines["bottom"].set_color("#bbbbbb")
        ax.set_facecolor("white")
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#dddddd", zorder=0)
        ax.set_axisbelow(True)

        # block labels
        lkw = dict(transform=ax.transAxes, fontsize=9, va="top", fontweight="bold")
        ax.text(240/2/MAX_T,         0.96, "Video 1",          ha="center", color=C_VIDEO_LN, **lkw)
        ax.text((240+324)/2/MAX_T,   0.96, "Waiting\nPeriod 1",ha="center", color=C_WAIT_LN,  **lkw)
        ax.text((324+564)/2/MAX_T,   0.96, "Video 2",          ha="center", color=C_VIDEO_LN, **lkw)
        ax.text((564+MAX_T)/2/MAX_T, 0.96, "Waiting\nPeriod 2",ha="center", color=C_WAIT_LN,  **lkw)

    axes[2].set_xlabel("Time (seconds)", fontsize=10, color=NAVY)

    # legend below
    legend_handles = [
        mpatches.Patch(facecolor=C_VIDEO_BG, edgecolor=C_VIDEO_LN, linewidth=1.2,
                       label="Video viewing block (Shamshiri et al., 2016)"),
        mpatches.Patch(facecolor=C_WAIT_BG,  edgecolor=C_WAIT_LN,  linewidth=1.2,
                       label="Waiting period (Shamshiri et al., 2016)"),
        plt.Line2D([0], [0], color=C_VIDEO_LN, linewidth=2,
                   label="DySCo signal — video block"),
        plt.Line2D([0], [0], color=C_WAIT_LN,  linewidth=2,
                   label="DySCo signal — waiting period"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2,
               fontsize=9, frameon=True, framealpha=0.9, edgecolor="#cccccc",
               bbox_to_anchor=(0.5, -0.04))

    run_label = f"{n_runs} cartoon run{'s' if n_runs > 1 else ''}"
    fig.suptitle(
        f"{pid}  —  Within-Cartoon DySCo Metrics\n"
        f"Average across {run_label}  |  Block structure from Shamshiri et al. (2016)",
        fontsize=13, fontweight="bold", color=NAVY, y=1.01
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out_path = os.path.join(OUT, f"fig_within_cartoon_{pnum}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")

print("\nDone — all 5 patients saved.")
