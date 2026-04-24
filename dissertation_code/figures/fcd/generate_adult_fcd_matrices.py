"""
generate_adult_fcd_matrices.py
================================
Group-average FCD matrices for the adult epilepsy dataset (n=10).
Produces four figures mirroring the paediatric FCD analysis:
  1. Overall cartoon FCD (all windows)
  2. Video-only FCD (windows in video blocks)
  3. Wait-only FCD (windows in wait blocks)
  4. Rest FCD (all windows from rest runs)

Saved to:
  D:/encrypt_generalised_adult/ADULT/dysco_results/group_average/fcd_matrices/
"""

import os
import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── CONFIG ────────────────────────────────────────────────────────────────────
ADULT_ROOT = r"D:\encrypt_generalised_adult\ADULT\dysco_results"
OUT_DIR    = os.path.join(ADULT_ROOT, "group_average", "fcd_matrices")
os.makedirs(OUT_DIR, exist_ok=True)

HALF_WIN = 10
LAG      = 20
TR       = 2.16

VIDEO_BLOCKS_VOL = [(0, 111), (150, 261)]
WAIT_BLOCKS_VOL  = [(111, 150), (261, 300)]

NAVY = "#1a2e4a"

# ── helpers ───────────────────────────────────────────────────────────────────

def _is_cartoon(fname):
    return "task-cartoon" in fname

def _is_rest(fname):
    return "task-rest" in fname

def _video_idx(T):
    return [i for i in range(T)
            if any(s <= i + HALF_WIN < e for s, e in VIDEO_BLOCKS_VOL)]

def _wait_idx(T):
    return [i for i in range(T)
            if any(s <= i + HALF_WIN < e for s, e in WAIT_BLOCKS_VOL)]


def _submatrix(fcd, idx):
    """Extract sub-FCD for a subset of window indices."""
    idx = np.array(idx)
    return fcd[np.ix_(idx, idx)]


def _offdiag(mat):
    n = mat.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return mat[mask]


# ── load and accumulate FCD matrices ─────────────────────────────────────────
cartoon_mats  = []   # full cartoon FCD (per run)
video_mats    = []   # video-block sub-FCD
wait_mats     = []   # wait-block sub-FCD
rest_mats     = []   # full rest FCD (per run)

pid_dirs = sorted(
    d for d in os.listdir(ADULT_ROOT)
    if d.startswith("sub-ga") and d.endswith("_dysco_output")
    and os.path.isdir(os.path.join(ADULT_ROOT, d))
)

for dname in pid_dirs:
    folder = os.path.join(ADULT_ROOT, dname)
    # ses-01 only for consistency
    npy_files = [f for f in glob.glob(os.path.join(folder, "*_dysco.npy"))
                 if "ses-01" in os.path.basename(f)]

    for fpath in npy_files:
        fname = os.path.basename(fpath)
        try:
            d = np.load(fpath, allow_pickle=True).item()
        except Exception as e:
            print(f"  Could not load {fname}: {e}")
            continue

        if "fcd" not in d:
            print(f"  No FCD in {fname} — skipping")
            continue

        fcd = np.asarray(d["fcd"], dtype=float)
        T   = fcd.shape[0]

        if _is_cartoon(fname):
            cartoon_mats.append(fcd)
            vid = _video_idx(T)
            wt  = _wait_idx(T)
            if vid:
                video_mats.append(_submatrix(fcd, vid))
            if wt:
                wait_mats.append(_submatrix(fcd, wt))
        elif _is_rest(fname):
            rest_mats.append(fcd)

print(f"Cartoon runs: {len(cartoon_mats)},  Video sub: {len(video_mats)},  "
      f"Wait sub: {len(wait_mats)},  Rest runs: {len(rest_mats)}")


# ── group average helper ──────────────────────────────────────────────────────

def _group_avg(mats):
    """Average FCD matrices, truncating to the smallest common size."""
    if not mats:
        return None
    min_T = min(m.shape[0] for m in mats)
    stack = np.array([m[:min_T, :min_T] for m in mats])
    return stack.mean(axis=0)


# ── plotting helper ───────────────────────────────────────────────────────────

def _plot_fcd(avg_fcd, title, fname, tr=TR, half_win=HALF_WIN, vmin=None, vmax=None):
    if avg_fcd is None:
        print(f"  Skipping {fname} — no data")
        return

    T      = avg_fcd.shape[0]
    t_axis = (np.arange(T) + half_win) * tr   # seconds

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("white")

    im = ax.imshow(avg_fcd, origin="lower", aspect="auto",
                   cmap="viridis", vmin=vmin, vmax=vmax,
                   extent=[t_axis[0], t_axis[-1], t_axis[0], t_axis[-1]])

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("DySCo Distance", fontsize=10, color=NAVY)
    cbar.ax.tick_params(labelsize=8, colors=NAVY)

    ax.set_xlabel("Time (seconds)", fontsize=10, color=NAVY)
    ax.set_ylabel("Time (seconds)", fontsize=10, color=NAVY)
    ax.set_title(title, fontsize=12, fontweight="bold", color=NAVY, pad=10)
    ax.tick_params(labelcolor=NAVY, labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # annotate n
    n_runs = len(cartoon_mats) if "Cartoon" in title or "Video" in title or "Wait" in title \
             else len(rest_mats)
    ax.text(0.98, 0.02, f"n = {n_runs} runs\n(10 participants, ses-01)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#555555", style="italic")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


# ── also make a 4-panel summary figure ───────────────────────────────────────

def _plot_4panel(avgs, titles, out_path, tr=TR, half_win=HALF_WIN, vmin=None, vmax=None):
    fig, axes = plt.subplots(2, 2, figsize=(13, 11),
                             gridspec_kw={"hspace": 0.35, "wspace": 0.28})
    fig.patch.set_facecolor("white")
    axes_flat = axes.flatten()

    last_im = None
    for ax, avg, title in zip(axes_flat, avgs, titles):
        if avg is None:
            ax.set_visible(False)
            continue
        T = avg.shape[0]
        t = (np.arange(T) + half_win) * tr
        last_im = ax.imshow(avg, origin="lower", aspect="auto", cmap="viridis",
                            vmin=vmin, vmax=vmax,
                            extent=[t[0], t[-1], t[0], t[-1]])
        ax.set_title(title, fontsize=12, fontweight="bold", color=NAVY, pad=10)
        ax.set_xlabel("Time (seconds)", fontsize=10, color=NAVY)
        ax.set_ylabel("Time (seconds)", fontsize=10, color=NAVY)
        ax.tick_params(labelcolor=NAVY, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#bbbbbb")
            spine.set_linewidth(0.8)

    # colorbar outside all panels, no overlap
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.12, 0.018, 0.74])
    cbar = fig.colorbar(last_im, cax=cbar_ax)
    cbar.set_label("DySCo Distance", fontsize=11, color=NAVY, labelpad=12)
    cbar.ax.tick_params(labelcolor=NAVY, labelsize=9)
    cbar.outline.set_edgecolor("#bbbbbb")

    fig.suptitle(
        "Group Average FCD Matrices — Adult Generalised Epilepsy  (n = 10, ses-01)",
        fontsize=14, fontweight="bold", color=NAVY, y=0.98
    )
    fig.text(
        0.44, 0.01,
        "Low (blue/purple) = similar connectivity states  ·  High (yellow) = dissimilar  |  "
        "Colour scale matched across all panels",
        ha="center", fontsize=9, color="#777777", style="italic"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {os.path.basename(out_path)}")


# ── generate ──────────────────────────────────────────────────────────────────
avg_cartoon = _group_avg(cartoon_mats)
avg_video   = _group_avg(video_mats)
avg_wait    = _group_avg(wait_mats)
avg_rest    = _group_avg(rest_mats)

# shared colour scale: 5th/95th percentile of off-diagonal values across all four matrices
all_offdiag = np.concatenate([
    _offdiag(m) for m in [avg_cartoon, avg_video, avg_wait, avg_rest]
    if m is not None
])
g_vmin = np.percentile(all_offdiag, 5)
g_vmax = np.percentile(all_offdiag, 95)
print(f"Global colour scale: vmin={g_vmin:.4f}, vmax={g_vmax:.4f}")

_plot_fcd(avg_cartoon, "Group Average FCD — Cartoon (all windows)",
          "adult_fcd_cartoon_overall.png", vmin=g_vmin, vmax=g_vmax)
_plot_fcd(avg_video,   "Group Average FCD — Video blocks only",
          "adult_fcd_video_only.png", half_win=0, vmin=g_vmin, vmax=g_vmax)
_plot_fcd(avg_wait,    "Group Average FCD — Please Wait blocks only",
          "adult_fcd_wait_only.png", half_win=0, vmin=g_vmin, vmax=g_vmax)
_plot_fcd(avg_rest,    "Group Average FCD — Rest (all windows)",
          "adult_fcd_rest_overall.png", vmin=g_vmin, vmax=g_vmax)

_plot_4panel(
    [avg_cartoon, avg_video, avg_wait, avg_rest],
    ["Cartoon (all)", "Video blocks", "Please Wait blocks", "Rest (all)"],
    os.path.join(OUT_DIR, "adult_fcd_4panel.png"),
    vmin=g_vmin, vmax=g_vmax
)

print("\nDone.")
