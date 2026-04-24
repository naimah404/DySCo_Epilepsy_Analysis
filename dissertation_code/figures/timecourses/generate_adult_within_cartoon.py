"""
generate_adult_within_cartoon.py
==================================
Within-cartoon DySCo timecourse for all adult participants.
Mirrors the paediatric generate_within_cartoon_all_patients.py figure style.

3 panels (entropy, speed, norm2), averaged across a participant's ses-01
cartoon runs, with Shamshiri et al. (2016) block overlay.
Line colour switches between video (blue) and waiting period (orange-red).

Saved to:
  D:/encrypt_generalised_adult/ADULT/dysco_results/group_average/within_cartoon_per_patient/
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import uniform_filter1d

# ── CONFIG ────────────────────────────────────────────────────────────────────
ADULT_ROOT = r"D:\encrypt_generalised_adult\ADULT\dysco_results"
OUT_DIR    = os.path.join(ADULT_ROOT, "group_average", "within_cartoon_per_patient")
os.makedirs(OUT_DIR, exist_ok=True)

TR       = 2.16
HALF_WIN = 10
LAG      = 20
SMOOTH   = 5

VIDEO_BLOCKS = [(0, 240), (324, 564)]
WAIT_BLOCKS  = [(240, 324), (564, 648)]      # adult Wait 2 ends at 300×2.16=648s

# x-axis: last window centre = (279 + 10) × 2.16 ≈ 624s
MAX_T_EN  = (279 + HALF_WIN) * TR            # entropy/norm2 (280 windows)
MAX_T_SPD = (259 + HALF_WIN + LAG) * TR      # speed (260 windows, +LAG offset)

NAVY       = "#1a2e4a"
C_VIDEO_BG = "#d1e5f0"
C_WAIT_BG  = "#fddbc7"
C_VIDEO_LN = "#2166ac"
C_WAIT_LN  = "#d6604d"

# ── window condition assignment ───────────────────────────────────────────────

VIDEO_VOL = [(0, 111), (150, 261)]
WAIT_VOL  = [(111, 150), (261, 300)]

def _segment_colour(t_sec):
    """Return line colour based on time in seconds."""
    for s, e in VIDEO_BLOCKS:
        if s <= t_sec < e:
            return C_VIDEO_LN
    return C_WAIT_LN


def _plot_participant(pid_short, cartoon_files):
    en_list, spd_list, n2_list = [], [], []
    for fpath in cartoon_files:
        try:
            d = np.load(fpath, allow_pickle=True).item()
        except Exception as e:
            print(f"  Could not load {os.path.basename(fpath)}: {e}")
            continue
        if "entropy" in d: en_list.append(np.asarray(d["entropy"], dtype=float))
        if "speed"   in d: spd_list.append(np.asarray(d["speed"],   dtype=float))
        if "norm2"   in d: n2_list.append(np.asarray(d["norm2"],    dtype=float))

    if not en_list:
        print(f"  [{pid_short}] No data — skipping")
        return

    # average across runs (truncate to shortest)
    def _avg(lst):
        ml = min(len(x) for x in lst)
        return np.mean([x[:ml] for x in lst], axis=0)

    en_m  = _avg(en_list)
    n2_m  = _avg(n2_list) if n2_list else None
    spd_m = _avg(spd_list) if spd_list else None

    t_en  = (np.arange(len(en_m))  + HALF_WIN)        * TR
    t_spd = (np.arange(len(spd_m)) + HALF_WIN + LAG)  * TR if spd_m is not None else None

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=False)
    fig.patch.set_facecolor("white")

    panels = [
        (axes[0], t_en,  en_m,  "Von Neumann Entropy"),
        (axes[1], t_spd, spd_m, "Reconfiguration Speed"),
        (axes[2], t_en,  n2_m,  "Connectivity Norm (L2)"),
    ]

    for ax, t, sig, ylabel in panels:
        if sig is None or t is None:
            ax.set_visible(False)
            continue

        max_t = t[-1]
        sm    = uniform_filter1d(sig, size=SMOOTH, mode="nearest")

        ax.set_facecolor("white")

        # block shading
        for s, e in VIDEO_BLOCKS:
            ax.axvspan(s, min(e, max_t), color=C_VIDEO_BG, alpha=0.55, zorder=0)
        for s, e in WAIT_BLOCKS:
            ax.axvspan(s, min(e, max_t), color=C_WAIT_BG,  alpha=0.55, zorder=0)

        # line coloured by segment
        prev_c = _segment_colour(t[0])
        seg_t, seg_v = [t[0]], [sm[0]]
        for ti, vi in zip(t[1:], sm[1:]):
            c = _segment_colour(ti)
            if c == prev_c:
                seg_t.append(ti); seg_v.append(vi)
            else:
                ax.plot(seg_t, seg_v, color=prev_c, linewidth=1.8, zorder=2)
                seg_t, seg_v = [seg_t[-1], ti], [seg_v[-1], vi]
                prev_c = c
        ax.plot(seg_t, seg_v, color=prev_c, linewidth=1.8, zorder=2)

        # block labels
        lkw = dict(transform=ax.transAxes, fontsize=8.5, va="top", fontweight="bold")
        if max_t > 0:
            ax.text(min(240/2,  max_t/2)/max_t,         0.96, "Video 1", ha="center",
                    color=C_VIDEO_LN, **lkw)
            if max_t > 282:
                ax.text((240+324)/2/max_t, 0.96, "Waiting\nPeriod 1",  ha="center",
                        color=C_WAIT_LN, **lkw)
            if max_t > 444:
                ax.text((324+564)/2/max_t, 0.96, "Video 2", ha="center",
                        color=C_VIDEO_LN, **lkw)
            if max_t > 606:
                ax.text((564+max_t)/2/max_t, 0.96, "Waiting\nPeriod 2", ha="center",
                        color=C_WAIT_LN, **lkw)

        ax.set_ylabel(ylabel, fontsize=10, color=NAVY, labelpad=6)
        ax.set_xlim(0, max_t)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#bbbbbb")
        ax.spines["bottom"].set_color("#bbbbbb")
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#dddddd", zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(labelcolor=NAVY, labelsize=9)

    axes[2].set_xlabel("Time (seconds)", fontsize=10, color=NAVY)

    legend_handles = [
        mpatches.Patch(facecolor=C_VIDEO_BG, edgecolor=C_VIDEO_LN, linewidth=1.2,
                       label="Video block (Shamshiri et al., 2016)"),
        mpatches.Patch(facecolor=C_WAIT_BG,  edgecolor=C_WAIT_LN,  linewidth=1.2,
                       label="Waiting period (Shamshiri et al., 2016)"),
        plt.Line2D([0], [0], color=C_VIDEO_LN, linewidth=2, label="Video"),
        plt.Line2D([0], [0], color=C_WAIT_LN,  linewidth=2, label="Please Wait"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=9,
               frameon=True, framealpha=0.95, edgecolor="#cccccc",
               bbox_to_anchor=(0.5, -0.02))

    n_runs = len(cartoon_files)
    fig.suptitle(
        f"sub-{pid_short}  —  DySCo Metrics Within Cartoon Run\n"
        f"Average across {n_runs} ses-01 cartoon run(s)  |  "
        "Block structure from Shamshiri et al. (2016)",
        fontsize=13, fontweight="bold", color=NAVY, y=0.99
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.97])

    out_path = os.path.join(OUT_DIR, f"fig_within_cartoon_{pid_short}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(out_path)}")


# ── main ──────────────────────────────────────────────────────────────────────
pid_dirs = sorted(
    d for d in os.listdir(ADULT_ROOT)
    if d.startswith("sub-ga") and d.endswith("_dysco_output")
    and os.path.isdir(os.path.join(ADULT_ROOT, d))
)

for dname in pid_dirs:
    pid_short = dname.replace("sub-", "").replace("_dysco_output", "")
    folder    = os.path.join(ADULT_ROOT, dname)
    cartoon_files = sorted([
        f for f in glob.glob(os.path.join(folder, "*_dysco.npy"))
        if "task-cartoon" in os.path.basename(f) and "ses-01" in os.path.basename(f)
    ])
    if not cartoon_files:
        print(f"  [{pid_short}] No ses-01 cartoon files — skipping")
        continue
    print(f"  [{pid_short}] {len(cartoon_files)} ses-01 cartoon run(s)")
    _plot_participant(pid_short, cartoon_files)

print("\nDone.")
