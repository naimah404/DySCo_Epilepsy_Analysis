"""
DySCo NIfTI Pipeline
====================
End-to-end pipeline:  NIfTI / .npy  →  DySCo .npy  →  all figures.

HOW TO USE
----------
1. Scroll to the CONFIG block at the very bottom of this file (Section 8).
2. Fill in your paths, participants, and parameters.
3. Uncomment the run_pipeline(...) call at the bottom.
4. Run:  python dysco_nifti_pipeline.py

WHAT IT PRODUCES
----------------
Per participant (saved under <output_root>/<PID>_dysco_output/):
  <run_name>_dysco.npy          — dictionary of all DySCo measures

Figures (saved under <output_root>/pipeline_figures/):
  per_patient/
    <pid>_timecourse.png        — Three-panel time-course (entropy, speed, norm2)
  group/
    group_cartoon.png           — Group avg cartoon + Shamshiri block overlay
    group_rest.png              — Group avg resting-state time-course
    group_wait_aligned.png      — Waiting period onset-aligned group average
  boxplots/
    three_condition_boxplots.png — Video / Please Wait / Rest per metric
    group_metric_summary.png    — Group-averaged distributions: norm2, entropy, speed

PARADIGM ASSUMPTIONS (Shamshiri et al., 2016)
---------------------------------------------
Cartoon run block structure (volumes):
  Video 1 :  0 – 111
  Wait  1 : 111 – 150
  Video 2 : 150 – 261
  Wait  2 : 261 – 296

Rest runs: pure resting state, no internal block structure.

PROGRESS BAR
------------
Uses tqdm if available, otherwise falls back to a built-in ASCII bar.
Install tqdm with:  pip install tqdm
"""

import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for all environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import uniform_filter1d
import nibabel as nb

# ── optional tqdm (graceful fallback) ─────────────────────────────────────────
try:
    from tqdm import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

# ── DySCo core functions ───────────────────────────────────────────────────────
# Adjust _CORE_PATH if your core_functions directory is located elsewhere.
# The default assumes:  <repo_root>/Python/core_functions/
_CORE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "core_functions")
)
if _CORE_PATH not in sys.path:
    sys.path.append(_CORE_PATH)

from compute_eigenvectors_sliding_cov import compute_eigs_cov
from dysco_distance import dysco_distance
from dysco_norm import dysco_norm


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — PROGRESS BAR HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _ascii_bar(current, total, width=40, label=""):
    """Simple ASCII progress bar used when tqdm is not installed."""
    frac  = current / max(total, 1)
    filled = int(width * frac)
    bar   = "=" * filled + ">" + " " * (width - filled - 1)
    pct   = frac * 100
    print(f"\r  [{bar}] {pct:5.1f}%  ({current}/{total})  {label}",
          end="", flush=True)
    if current >= total:
        print()   # newline when done


def _make_progress(iterable, total=None, desc=""):
    """
    Wrap an iterable with a progress bar.
    Uses tqdm if available, otherwise prints an ASCII bar after each item.
    """
    if _TQDM_AVAILABLE:
        return _tqdm(iterable, total=total, desc=desc, ncols=80,
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]")
    # fallback: wrap in a generator that prints after each step
    n = total if total is not None else len(iterable)
    def _gen():
        for i, item in enumerate(iterable, 1):
            yield item
            _ascii_bar(i, n, label=desc)
    return _gen()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PROCESSING  (NIfTI / .npy  →  DySCo .npy)
# ══════════════════════════════════════════════════════════════════════════════

def process_single_file(file_path, output_folder,
                        half_window_size=10, n_eigen=10, lag=20,
                        save_slim=False):
    """
    Load one NIfTI (.nii / .nii.gz) or .npy fMRI file, run the full DySCo
    pipeline, and save a .npy dictionary containing all measures.

    Measures saved
    --------------
    norm1        : L1 norm of eigenvalue spectrum at each window
    norm2        : L2 norm (Frobenius) — primary DySCo metric
    norminf      : L-infinity norm
    metastability: std(norm2) across the run
    speed        : reconfiguration speed (off-diagonal FCD, lag windows apart)
    entropy      : von Neumann entropy of the eigenvalue spectrum
    fcd          : full T×T FCD (functional connectivity dynamics) matrix
    eigenvalues  : [n_eigen × T] eigenvalue matrix
    eigenvectors : [T × n_eigen × n_voxels] eigenvector tensor

    Parameters
    ----------
    file_path        : str   — path to .nii, .nii.gz, or .npy input file
    output_folder    : str   — directory where the _dysco.npy file is written
    half_window_size : int   — half the sliding-window width (total = 2*h+1)
    n_eigen          : int   — number of eigenvectors to retain
    lag              : int   — volume lag used for reconfiguration speed

    Returns
    -------
    dict  — all computed DySCo measures
    """
    os.makedirs(output_folder, exist_ok=True)

    # ── load ──────────────────────────────────────────────────────────────────
    if file_path.endswith(".nii") or file_path.endswith(".nii.gz"):
        nifti    = nb.load(file_path)
        fmri_4d  = nifti.get_fdata()
        # reshape 4-D [X, Y, Z, T] → 2-D [T, voxels]
        brain_2d = fmri_4d.reshape(-1, fmri_4d.shape[-1]).T
    elif file_path.endswith(".npy"):
        brain_2d = np.load(file_path)        # expected shape: [T, voxels]
    else:
        raise ValueError(
            f"Unsupported format for '{file_path}'. "
            "Accepted: .nii, .nii.gz, .npy"
        )

    brain_2d = np.nan_to_num(brain_2d).astype(np.float64)
    brain_2d += 1e-6 * np.random.randn(*brain_2d.shape)   # regularisation

    # ── eigenvectors & eigenvalues ─────────────────────────────────────────────
    eigenvectors, eigenvalues = compute_eigs_cov(
        brain_2d, n_eigen, half_window_size
    )
    T = eigenvectors.shape[0]   # number of sliding windows

    # ── norms ─────────────────────────────────────────────────────────────────
    norm1   = dysco_norm(eigenvalues, 1)
    norm2   = dysco_norm(eigenvalues, 2)
    norminf = dysco_norm(eigenvalues, np.inf)

    # ── metastability (temporal variability of norm2) ─────────────────────────
    metastability = float(np.std(norm2))

    # ── FCD matrix  [T × T] ───────────────────────────────────────────────────
    fcd = np.zeros((T, T))
    for i in range(T):
        for j in range(i + 1, T):
            fcd[i, j] = dysco_distance(eigenvectors[i], eigenvectors[j], 2)
            fcd[j, i] = fcd[i, j]

    # ── reconfiguration speed ─────────────────────────────────────────────────
    speed = np.array([fcd[i, i + lag] for i in range(T - lag)])

    # ── von Neumann entropy ────────────────────────────────────────────────────
    ev_norm = eigenvalues / (np.sum(eigenvalues, axis=0, keepdims=True) + 1e-10)
    entropy = -np.sum(ev_norm * np.log(ev_norm + 1e-10), axis=0)

    # ── bundle into a dictionary ───────────────────────────────────────────────
    measures = {
        "filename":      os.path.basename(file_path),
        "norm1":         norm1,
        "norm2":         norm2,          # primary DySCo metric
        "norminf":       norminf,
        "metastability": metastability,
        "speed":         speed,          # reconfiguration speed
        "entropy":       entropy,        # von Neumann entropy
        "fcd":           fcd,
        "eigenvalues":   eigenvalues,
        "eigenvectors":  eigenvectors,
    }

    # ── save ───────────────────────────────────────────────────────────────────
    # Strip both extensions to handle .nii.gz correctly
    base     = os.path.splitext(os.path.splitext(os.path.basename(file_path))[0])[0]
    out_path = os.path.join(output_folder, f"{base}_dysco.npy")
    # save_slim drops eigenvectors (3+ GB each) — use when storage is limited
    to_save  = {k: v for k, v in measures.items() if k != "eigenvectors"} \
               if save_slim else measures
    np.save(out_path, to_save)
    # free eigenvectors from RAM immediately after saving — they are 3+ GB each
    if save_slim and "eigenvectors" in measures:
        del measures["eigenvectors"]
    return measures, out_path


def batch_process_participant(pid, input_paths, output_folder,
                               half_window_size=10, n_eigen=10, lag=20,
                               show_progress=True, save_slim=False,
                               skip_existing=True):
    """
    Process all fMRI files for one participant and return a list of
    (run_id, measures) tuples.

    Parameters
    ----------
    pid            : str        — participant label, e.g. "P001"
    input_paths    : str | list — folder path (glob'd) OR explicit file list
    output_folder  : str        — where _dysco.npy files are saved
    show_progress  : bool       — show per-file progress bar
    skip_existing  : bool       — if True, skip files whose _dysco.npy already
                                  exists (>100 KB) and load them instead
    """
    os.makedirs(output_folder, exist_ok=True)

    # resolve to a flat list of file paths
    if isinstance(input_paths, (list, tuple)):
        file_list = list(input_paths)
    else:
        file_list = sorted(
            glob.glob(os.path.join(input_paths, "*.nii")) +
            glob.glob(os.path.join(input_paths, "*.nii.gz")) +
            glob.glob(os.path.join(input_paths, "*.npy"))
        )

    if not file_list:
        print(f"  [{pid}] No files found in: {input_paths}")
        return []

    results = []
    iterator = (
        _make_progress(file_list, total=len(file_list), desc=f"  {pid}")
        if show_progress else file_list
    )

    for fpath in iterator:
        base   = os.path.splitext(os.path.splitext(os.path.basename(fpath))[0])[0]
        run_id = base
        npy_path = os.path.join(output_folder, f"{base}_dysco.npy")

        if skip_existing and os.path.exists(npy_path) and os.path.getsize(npy_path) > 100_000:
            try:
                measures = np.load(npy_path, allow_pickle=True).item()
                results.append((run_id, measures))
                print(f"    Skipped (exists): {os.path.basename(npy_path)}")
                continue
            except Exception:
                pass   # fall through to reprocess if load fails

        try:
            print(f"\n  >>> Processing: {os.path.basename(fpath)}", flush=True)
            measures, out_path = process_single_file(
                fpath, output_folder, half_window_size, n_eigen, lag,
                save_slim=save_slim
            )
            results.append((run_id, measures))
            print(f"  <<< Done: {os.path.basename(out_path)}", flush=True)
        except Exception as exc:
            print(f"\n    ERROR processing {fpath}: {exc}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SHARED STYLE CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

NAVY        = "#1a2e4a"
C_CARTOON   = "#2166ac"
C_WAIT_LN   = "#d6604d"
C_WAIT_LBL  = "#b35806"
C_REST      = "#d6604d"
C_VIDEO_BG  = "#d1e5f0"
C_WAIT_BG   = "#fddbc7"
C_REST_FILL = "#fddbc7"

# Shamshiri et al. (2016) block boundaries
VIDEO_BLOCKS_VOL = [(0, 111), (150, 261)]   # volume indices
WAIT_BLOCKS_VOL  = [(111, 150), (261, 296)]
VIDEO_BLOCKS_SEC = [(0, 240), (324, 564)]    # seconds
WAIT_BLOCKS_SEC  = [(240, 324), (564, 639.4)]

# DySCo sliding-window defaults — must match what was used during processing
HALF_WIN     = 10
LAG          = 20
TR_DEFAULT   = 2.16
SMOOTH       = 5       # uniform-filter smoothing width (windows)
N_WIN        = 276     # expected number of windows per cartoon run
WAIT_WIN_LEN = 25      # windows extracted per waiting period for onset alignment

# Window index sets for block-level extraction (Shamshiri volumes)
_vid_idx, _wait_idx = [], []
for _k in range(N_WIN):
    _centre = _k + HALF_WIN
    if any(s <= _centre < e for s, e in VIDEO_BLOCKS_VOL):
        _vid_idx.append(_k)
    elif any(s <= _centre < e for s, e in WAIT_BLOCKS_VOL):
        _wait_idx.append(_k)
VID_IDX  = np.array(_vid_idx)    # ~212 windows
WAIT_IDX = np.array(_wait_idx)   # ~64 windows

# Window index ranges for onset-aligned waiting-period extraction
WAIT_EN_RANGES  = [(101, 126), (251, 276)]   # entropy / norm2
WAIT_SPD_RANGES = [(81,  106), (231, 256)]   # speed (shifted by LAG)

METRICS = [
    ("entropy", "Von Neumann Entropy"),
    ("speed",   "Reconfiguration Speed"),
    ("norm2",   "Connectivity Norm (L2)"),
]


# ── shared axis styling ────────────────────────────────────────────────────────
def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#bbbbbb")
    ax.spines["bottom"].set_color("#bbbbbb")
    ax.set_facecolor("white")
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#dddddd", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelcolor=NAVY, labelsize=9)


def _block_labels(ax, max_t):
    """Add Video/Wait text labels along the top of a cartoon-run axis."""
    lkw = dict(transform=ax.transAxes, fontsize=8.5, va="top", fontweight="bold")
    ax.text(240/2/max_t,         0.96, "Video 1", ha="center", color=C_CARTOON,  **lkw)
    ax.text((240+324)/2/max_t,   0.96, "Waiting Period 1",  ha="center", color=C_WAIT_LBL, **lkw)
    ax.text((324+564)/2/max_t,   0.96, "Video 2",           ha="center", color=C_CARTOON,  **lkw)
    ax.text((564+max_t)/2/max_t, 0.96, "Waiting Period 2",  ha="center", color=C_WAIT_LBL, **lkw)


def _avg(lst):
    """Average a list of arrays after truncating to the shortest length."""
    ml = min(len(x) for x in lst)
    return np.mean([x[:ml] for x in lst], axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PER-PATIENT FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def plot_patient_timecourse(pid, run_data, out_dir, TR=TR_DEFAULT):
    """
    Three-panel smoothed time-course (entropy, speed, norm2) for one patient.
    All available runs (cartoon + rest) are averaged together.

    Parameters
    ----------
    pid      : str   — participant label, e.g. "P001"
    run_data : list  — [(run_id, measures_dict), ...] — all run types pooled
    out_dir  : str   — root output directory (sub-folder per_patient/ is created)
    TR       : float — repetition time in seconds

    Saved to
    --------
    <out_dir>/per_patient/<pid_lower>_timecourse.png
    """
    os.makedirs(os.path.join(out_dir, "per_patient"), exist_ok=True)

    en_all, spd_all, n2_all = [], [], []
    for _, m in run_data:
        en_all.append(np.array(m["entropy"]))
        spd_all.append(np.array(m["speed"]))
        n2_all.append(np.array(m["norm2"]))

    en_m  = _avg(en_all)
    spd_m = _avg(spd_all)
    n2_m  = _avg(n2_all)

    t_en  = (np.arange(len(en_m))  + HALF_WIN)       * TR
    t_spd = (np.arange(len(spd_m)) + HALF_WIN + LAG) * TR
    max_t = t_en[-1]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    fig.patch.set_facecolor("white")

    panels = [
        (axes[0], t_en,  uniform_filter1d(en_m,  SMOOTH), "Von Neumann Entropy"),
        (axes[1], t_spd, uniform_filter1d(spd_m, SMOOTH), "Reconfiguration Speed"),
        (axes[2], t_en,  uniform_filter1d(n2_m,  SMOOTH), "Connectivity Norm (L2)"),
    ]
    for ax, t, sig, ylabel in panels:
        ax.plot(t, sig, color=C_CARTOON, linewidth=1.8, zorder=2)
        ax.set_ylabel(ylabel, fontsize=10, color=NAVY, labelpad=6)
        ax.set_xlim(0, max_t)
        _style_ax(ax)

    axes[2].set_xlabel("Time (seconds)", fontsize=10, color=NAVY)
    fig.suptitle(
        f"{pid}  —  DySCo Metrics Time-Course\n"
        f"Average across {len(run_data)} run(s)",
        fontsize=13, fontweight="bold", color=NAVY, y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, "per_patient", f"{pid.lower()}_timecourse.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — GROUP FIGURE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _group_stats(all_run_data_by_pid, condition_key, metric):
    """
    Compute group mean ± SD time-course for a given condition and metric.

    Parameters
    ----------
    all_run_data_by_pid : dict — {pid: {"cartoon": [...], "rest": [...]}}
    condition_key       : str  — "cartoon" or "rest"
    metric              : str  — key in the measures dictionary

    Returns
    -------
    (mean_array, sd_array)  or  (None, None) if no data available
    """
    per_part = []
    for pid, cond_dict in all_run_data_by_pid.items():
        tcs = [np.array(m[metric]) for _, m in cond_dict.get(condition_key, [])]
        if tcs:
            ml = min(len(t) for t in tcs)
            per_part.append(np.mean([t[:ml] for t in tcs], axis=0))
    if not per_part:
        return None, None
    ml  = min(len(p) for p in per_part)
    arr = np.array([p[:ml] for p in per_part])
    return arr.mean(axis=0), arr.std(axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — GROUP CARTOON  (Shamshiri block overlay)
# ══════════════════════════════════════════════════════════════════════════════

def plot_group_cartoon(all_run_data_by_pid, out_dir, TR=TR_DEFAULT):
    """
    Group average cartoon time-course with Shamshiri et al. block overlay.

    Three panels: entropy, reconfiguration speed, norm2.
    Group mean ± SD shaded band.  Block shading from Shamshiri et al. (2016).

    Parameters
    ----------
    all_run_data_by_pid : dict  — {pid: {"cartoon": [...], "rest": [...]}}
    out_dir             : str   — root output directory (group/ sub-folder created)
    TR                  : float — repetition time in seconds

    Saved to
    --------
    <out_dir>/group/group_cartoon.png
    """
    os.makedirs(os.path.join(out_dir, "group"), exist_ok=True)
    n_part = len(all_run_data_by_pid)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    fig.patch.set_facecolor("white")

    for ax, (metric, ylabel) in zip(axes, METRICS):
        mean, sd = _group_stats(all_run_data_by_pid, "cartoon", metric)
        if mean is None:
            continue
        t     = (np.arange(len(mean)) + HALF_WIN + (LAG if metric == "speed" else 0)) * TR
        max_t = t[-1]

        # block shading
        for s, e in VIDEO_BLOCKS_SEC:
            ax.axvspan(s, min(e, max_t), color=C_VIDEO_BG, alpha=0.55, zorder=0)
        for s, e in WAIT_BLOCKS_SEC:
            ax.axvspan(s, min(e, max_t), color=C_WAIT_BG,  alpha=0.55, zorder=0)

        sm   = uniform_filter1d(mean, SMOOTH)
        sm_s = uniform_filter1d(sd,   SMOOTH)
        ax.fill_between(t, sm - sm_s, sm + sm_s, color=C_VIDEO_BG, alpha=0.45, zorder=1)
        ax.plot(t, sm, color=C_CARTOON, linewidth=2, zorder=2)
        _block_labels(ax, max_t)
        ax.set_ylabel(ylabel, fontsize=10, color=NAVY, labelpad=6)
        ax.set_xlim(0, max_t)
        _style_ax(ax)

    axes[2].set_xlabel("Time (seconds)", fontsize=10, color=NAVY)
    legend_handles = [
        mpatches.Patch(facecolor=C_VIDEO_BG, edgecolor=C_CARTOON, linewidth=1.2,
                       label="Video block (Shamshiri et al., 2016)"),
        mpatches.Patch(facecolor=C_WAIT_BG,  edgecolor=C_WAIT_LBL, linewidth=1.2,
                       label="Waiting period (Shamshiri et al., 2016)"),
        plt.Line2D([0], [0], color=C_CARTOON, linewidth=2,
                   label=f"Cartoon — group mean ± SD  (n = {n_part})"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=9,
               frameon=True, framealpha=0.95, edgecolor="#cccccc",
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f"Group Average DySCo Metrics — Cartoon Condition  (n = {n_part})\n"
        "Block structure from Shamshiri et al. (2016)",
        fontsize=13, fontweight="bold", color=NAVY, y=0.98
    )
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    path = os.path.join(out_dir, "group", "group_cartoon.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — GROUP RESTING STATE
# ══════════════════════════════════════════════════════════════════════════════

def plot_group_rest(all_run_data_by_pid, out_dir, TR=TR_DEFAULT):
    """
    Group average resting-state time-course.  Three panels: entropy, speed, norm2.

    Parameters
    ----------
    all_run_data_by_pid : dict  — {pid: {"cartoon": [...], "rest": [...]}}
    out_dir             : str   — root output directory
    TR                  : float — repetition time in seconds

    Saved to
    --------
    <out_dir>/group/group_rest.png
    """
    os.makedirs(os.path.join(out_dir, "group"), exist_ok=True)
    n_part = len(all_run_data_by_pid)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    fig.patch.set_facecolor("white")

    for ax, (metric, ylabel) in zip(axes, METRICS):
        mean, sd = _group_stats(all_run_data_by_pid, "rest", metric)
        if mean is None:
            continue
        t     = (np.arange(len(mean)) + HALF_WIN + (LAG if metric == "speed" else 0)) * TR
        max_t = t[-1]

        sm   = uniform_filter1d(mean, SMOOTH)
        sm_s = uniform_filter1d(sd,   SMOOTH)
        ax.fill_between(t, sm - sm_s, sm + sm_s, color=C_REST_FILL, alpha=0.45, zorder=1)
        ax.plot(t, sm, color=C_REST, linewidth=2, zorder=2)
        ax.set_ylabel(ylabel, fontsize=10, color=NAVY, labelpad=6)
        ax.set_xlim(0, max_t)
        _style_ax(ax)

    axes[2].set_xlabel("Time (seconds)", fontsize=10, color=NAVY)
    legend_handles = [
        plt.Line2D([0], [0], color=C_REST, linewidth=2,
                   label=f"Rest — group mean ± SD  (n = {n_part})"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=1, fontsize=9,
               frameon=True, framealpha=0.95, edgecolor="#cccccc",
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f"Group Average DySCo Metrics — Resting State  (n = {n_part})",
        fontsize=13, fontweight="bold", color=NAVY, y=0.98
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    path = os.path.join(out_dir, "group", "group_rest.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — WAITING PERIOD ONSET-ALIGNED
# ══════════════════════════════════════════════════════════════════════════════

def plot_group_wait_aligned(all_run_data_by_pid, out_dir, TR=TR_DEFAULT):
    """
    Waiting period onset-aligned group average.

    Both waiting periods from each cartoon run are extracted, aligned from
    their onset, averaged per participant, then the group mean ± SD is computed.

    Parameters
    ----------
    all_run_data_by_pid : dict  — {pid: {"cartoon": [...], "rest": [...]}}
    out_dir             : str   — root output directory
    TR                  : float — repetition time in seconds

    Saved to
    --------
    <out_dir>/group/group_wait_aligned.png
    """
    os.makedirs(os.path.join(out_dir, "group"), exist_ok=True)
    n_part = len(all_run_data_by_pid)

    C_WAIT_FILL = "#fee0b6"
    C_WAIT_LINE = "#b35806"

    # Each metric entry: (key, y-label, window-index ranges for extraction)
    metric_specs = [
        ("entropy", "Von Neumann Entropy",    WAIT_EN_RANGES),
        ("speed",   "Reconfiguration Speed",  WAIT_SPD_RANGES),
        ("norm2",   "Connectivity Norm (L2)", WAIT_EN_RANGES),
    ]

    t_wait = np.arange(WAIT_WIN_LEN) * TR   # seconds from period onset

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    fig.patch.set_facecolor("white")

    for ax, (metric, ylabel, ranges) in zip(axes, metric_specs):
        per_part = []
        for pid, cond_dict in all_run_data_by_pid.items():
            periods = []
            for _, m in cond_dict.get("cartoon", []):
                tc = np.array(m[metric])
                for s, e in ranges:
                    seg = tc[s:e]
                    if len(seg) >= WAIT_WIN_LEN:
                        periods.append(seg[:WAIT_WIN_LEN])
            if periods:
                per_part.append(np.mean(periods, axis=0))
        if not per_part:
            continue

        arr  = np.array(per_part)
        mean = arr.mean(axis=0)
        sd   = arr.std(axis=0)
        sm   = uniform_filter1d(mean, min(SMOOTH, WAIT_WIN_LEN // 3))
        sm_s = uniform_filter1d(sd,   min(SMOOTH, WAIT_WIN_LEN // 3))

        ax.fill_between(t_wait, sm - sm_s, sm + sm_s,
                        color=C_WAIT_FILL, alpha=0.55, zorder=1)
        ax.plot(t_wait, sm, color=C_WAIT_LINE, linewidth=2, zorder=2)
        ax.set_ylabel(ylabel, fontsize=10, color=NAVY, labelpad=6)
        ax.set_xlim(0, t_wait[-1])
        _style_ax(ax)

    axes[2].set_xlabel("Time from waiting period onset (seconds)",
                       fontsize=10, color=NAVY)
    legend_handles = [
        plt.Line2D([0], [0], color=C_WAIT_LINE, linewidth=2,
                   label=f"Waiting period — group mean ± SD  (n = {n_part})\n"
                         "(Waiting Period 1 & 2 averaged, aligned from onset)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=1, fontsize=9,
               frameon=True, framealpha=0.95, edgecolor="#cccccc",
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f"Group Average DySCo Metrics — Waiting Period Onset-Aligned Analysis  (n = {n_part})\n"
        "Waiting periods aligned from onset, averaged across both periods per run",
        fontsize=13, fontweight="bold", color=NAVY, y=0.98
    )
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    path = os.path.join(out_dir, "group", "group_wait_aligned.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — THREE-CONDITION BOXPLOT
# ══════════════════════════════════════════════════════════════════════════════

def plot_three_condition_boxplot(all_run_data_by_pid, out_dir):
    """
    Three-condition boxplot: Video  /  Please Wait  /  Rest.

    One column per metric (entropy, speed, norm2).
    Boxes = group-level window distributions.
    Dots  = individual participant medians overlaid with jitter.

    Block assignment (Shamshiri volumes):
      Video :  0–111, 150–261   (centre volume = window index + HALF_WIN)
      Wait  : 111–150, 261–296
      Rest  : all windows from rest runs

    Parameters
    ----------
    all_run_data_by_pid : dict — {pid: {"cartoon": [...], "rest": [...]}}
    out_dir             : str  — root output directory

    Saved to
    --------
    <out_dir>/boxplots/three_condition_boxplots.png
    """
    os.makedirs(os.path.join(out_dir, "boxplots"), exist_ok=True)

    C_WAIT_COL  = "#b35806"
    C_WAIT_FILL = "#fee0b6"
    CONDITIONS  = ["video", "wait", "rest"]
    EDGE   = {"video": C_CARTOON,   "wait": C_WAIT_COL,  "rest": C_REST}
    FILL   = {"video": C_VIDEO_BG,  "wait": C_WAIT_FILL, "rest": C_REST_FILL}
    DOT    = {"video": C_CARTOON,   "wait": C_WAIT_COL,  "rest": "#b2182b"}
    XLBLS  = {
        "video": "Video\n(cartoon)",
        "wait":  "Please wait\n(within cartoon)",
        "rest":  "Rest",
    }

    def _win_label_en(i):
        """Assign entropy/norm2 window i to a condition using its centre volume."""
        c = i + HALF_WIN
        if any(s <= c < e for s, e in VIDEO_BLOCKS_VOL): return "video"
        if any(s <= c < e for s, e in WAIT_BLOCKS_VOL):  return "wait"
        return None

    def _win_label_spd(i):
        """Assign speed window i to a condition (centre shifted by LAG)."""
        c = i + LAG + HALF_WIN
        if any(s <= c < e for s, e in VIDEO_BLOCKS_VOL): return "video"
        if any(s <= c < e for s, e in WAIT_BLOCKS_VOL):  return "wait"
        return None

    # accumulate window-level pools
    pools      = {m: {c: [] for c in CONDITIONS} for m, _ in METRICS}
    part_pools = {pid: {m: {c: [] for c in CONDITIONS} for m, _ in METRICS}
                  for pid in all_run_data_by_pid}

    for pid, cond_dict in all_run_data_by_pid.items():
        for _, m in cond_dict.get("cartoon", []):
            en  = np.array(m["entropy"])
            n2  = np.array(m["norm2"])
            spd = np.array(m["speed"])
            for i in range(len(en)):
                lbl = _win_label_en(i)
                if lbl:
                    for key, arr in [("entropy", en), ("norm2", n2)]:
                        pools[key][lbl].append(arr[i])
                        part_pools[pid][key][lbl].append(arr[i])
            for i in range(len(spd)):
                lbl = _win_label_spd(i)
                if lbl:
                    pools["speed"][lbl].append(spd[i])
                    part_pools[pid]["speed"][lbl].append(spd[i])

        for _, m in cond_dict.get("rest", []):
            for metric, _ in METRICS:
                vals = list(m[metric])
                pools[metric]["rest"].extend(vals)
                part_pools[pid][metric]["rest"].extend(vals)

    # participant-level medians
    part_medians = {
        metric: {
            c: [np.median(part_pools[pid][metric][c])
                for pid in all_run_data_by_pid
                if part_pools[pid][metric][c]]
            for c in CONDITIONS
        }
        for metric, _ in METRICS
    }
    n_counts = {metric: {c: len(pools[metric][c]) for c in CONDITIONS}
                for metric, _ in METRICS}

    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    fig.patch.set_facecolor("white")

    for ax, (metric, ylabel) in zip(axes, METRICS):
        data_by_cond = [np.array(pools[metric][c]) for c in CONDITIONS]
        bp = ax.boxplot(
            data_by_cond, positions=[1, 2, 3], widths=0.45,
            patch_artist=True, showfliers=True,
            medianprops=dict(color="white", linewidth=2.2, solid_capstyle="round"),
            whiskerprops=dict(linewidth=1.2, color="#555555"),
            capprops=dict(linewidth=1.2,   color="#555555"),
            flierprops=dict(marker="o", markersize=2.5, alpha=0.25,
                            markeredgewidth=0, linestyle="none"),
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

        for pos, cond in zip([1, 2, 3], CONDITIONS):
            meds   = part_medians[metric][cond]
            jitter = rng.uniform(-0.07, 0.07, size=len(meds))
            ax.scatter([pos + j for j in jitter], meds,
                       color=DOT[cond], s=40, zorder=6,
                       edgecolors="white", linewidths=0.8)

        tick_labels = [
            f"{XLBLS[c]}\n$n$ = {n_counts[metric][c]:,}" for c in CONDITIONS
        ]
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(tick_labels, fontsize=9, color=NAVY, linespacing=1.5)
        ax.set_xlim(0.4, 3.6)
        ax.set_ylabel(ylabel, fontsize=10.5, color=NAVY, labelpad=6)
        ax.set_title(ylabel, fontsize=12, fontweight="bold", color=NAVY, pad=10)
        ax.tick_params(axis="y", labelcolor=NAVY, labelsize=9)
        ax.tick_params(axis="x", length=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#bbbbbb")
        ax.spines["bottom"].set_color("#bbbbbb")
        ax.set_facecolor("white")
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#dddddd", zorder=0)
        ax.set_axisbelow(True)

    n_part = len(all_run_data_by_pid)
    fig.suptitle(
        "DySCo Metrics Across Conditions: Video, Please Wait, and Rest\n"
        f"(window-level, n = {n_part} participants)",
        fontsize=13, fontweight="bold", color=NAVY, y=1.02
    )
    fig.text(
        0.5, -0.04,
        "Boxes = group-level window distributions (IQR ± 1.5×IQR).  "
        "Dots = individual participant medians.",
        ha="center", fontsize=8, color="#555555", style="italic"
    )
    plt.tight_layout(rect=[0, 0, 1, 1])
    path = os.path.join(out_dir, "boxplots", "three_condition_boxplots.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — GROUP METRIC SUMMARY BOXPLOT  (norm2, entropy, recon speed)
# ══════════════════════════════════════════════════════════════════════════════

def plot_group_metric_summary(all_run_data_by_pid, out_dir):
    """
    Group-averaged review of all three DySCo metrics: norm2, entropy, speed.

    Each panel shows the distribution of a metric across ALL windows and ALL
    participants (cartoon + rest pooled), with individual participant medians
    overlaid as dots.  Useful as a high-level sanity check and metric comparison.

    Layout: 1 row × 3 columns (one column per metric).

    Parameters
    ----------
    all_run_data_by_pid : dict — {pid: {"cartoon": [...], "rest": [...]}}
    out_dir             : str  — root output directory

    Saved to
    --------
    <out_dir>/boxplots/group_metric_summary.png
    """
    os.makedirs(os.path.join(out_dir, "boxplots"), exist_ok=True)

    METRIC_COLS = {
        "entropy": {"fill": "#d1e5f0", "edge": "#2166ac", "dot": "#2166ac",
                    "label": "Von Neumann\nEntropy"},
        "speed":   {"fill": "#fddbc7", "edge": "#b35806", "dot": "#b35806",
                    "label": "Reconfiguration\nSpeed"},
        "norm2":   {"fill": "#d9f0d3", "edge": "#1b7837", "dot": "#1b7837",
                    "label": "Connectivity\nNorm (L2)"},
    }
    METRIC_ORDER = ["entropy", "speed", "norm2"]

    # collect all-windows pools and per-participant medians
    group_pools = {m: [] for m in METRIC_ORDER}   # flat list of all window values
    part_pools  = {pid: {m: [] for m in METRIC_ORDER}
                   for pid in all_run_data_by_pid}

    for pid, cond_dict in all_run_data_by_pid.items():
        for cond_key in ["cartoon", "rest"]:
            for _, m in cond_dict.get(cond_key, []):
                for metric in METRIC_ORDER:
                    vals = list(np.array(m[metric]))
                    group_pools[metric].extend(vals)
                    part_pools[pid][metric].extend(vals)

    # participant-level medians (one value per participant per metric)
    part_medians = {
        metric: [np.median(part_pools[pid][metric])
                 for pid in all_run_data_by_pid
                 if part_pools[pid][metric]]
        for metric in METRIC_ORDER
    }
    n_wins = {m: len(group_pools[m]) for m in METRIC_ORDER}

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    fig.patch.set_facecolor("white")

    for ax, metric in zip(axes, METRIC_ORDER):
        col  = METRIC_COLS[metric]
        data = [np.array(group_pools[metric])]

        bp = ax.boxplot(
            data, positions=[1], widths=0.5,
            patch_artist=True, showfliers=True,
            medianprops=dict(color="white", linewidth=2.5, solid_capstyle="round"),
            whiskerprops=dict(linewidth=1.3, color="#555555"),
            capprops=dict(linewidth=1.3,    color="#555555"),
            flierprops=dict(marker="o", markersize=2.5, alpha=0.2,
                            markeredgewidth=0, linestyle="none"),
            zorder=2,
        )
        bp["boxes"][0].set_facecolor(col["fill"])
        bp["boxes"][0].set_edgecolor(col["edge"])
        bp["boxes"][0].set_linewidth(1.8)
        bp["fliers"][0].set_markerfacecolor(col["edge"])
        bp["fliers"][0].set_markeredgecolor("none")

        # participant median dots
        meds   = part_medians[metric]
        jitter = rng.uniform(-0.10, 0.10, size=len(meds))
        ax.scatter([1 + j for j in jitter], meds,
                   color=col["dot"], s=55, zorder=6,
                   edgecolors="white", linewidths=1.0)

        # annotate n= and group median
        gmed = np.median(group_pools[metric])
        ax.text(1, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else 1,
                f"group median = {gmed:.4g}",
                ha="center", va="bottom", fontsize=8.5,
                color=col["edge"], transform=ax.get_xaxis_transform())

        ax.set_xticks([1])
        ax.set_xticklabels(
            [f"{col['label']}\n$n_{{win}}$ = {n_wins[metric]:,}"],
            fontsize=9.5, color=NAVY, linespacing=1.5
        )
        ax.set_xlim(0.3, 1.7)
        ax.set_ylabel(col["label"].replace("\n", " "), fontsize=10, color=NAVY, labelpad=6)
        ax.tick_params(axis="y", labelcolor=NAVY, labelsize=9)
        ax.tick_params(axis="x", length=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#bbbbbb")
        ax.spines["bottom"].set_color("#bbbbbb")
        ax.set_facecolor("white")
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#dddddd", zorder=0)
        ax.set_axisbelow(True)

    n_part = len(all_run_data_by_pid)
    fig.suptitle(
        f"Group Average DySCo Metric Distributions  (n = {n_part} participants)\n"
        "All conditions and runs pooled",
        fontsize=13, fontweight="bold", color=NAVY, y=1.02
    )
    fig.text(
        0.5, -0.04,
        "Boxes = group-level window distributions (IQR ± 1.5×IQR).  "
        "Dots = individual participant medians.  All runs pooled.",
        ha="center", fontsize=8, color="#555555", style="italic"
    )
    plt.tight_layout(rect=[0, 0, 1, 1])
    path = os.path.join(out_dir, "boxplots", "group_metric_summary.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — MASTER RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(cohort_config, output_root,
                 half_window_size=10, n_eigen=10, lag=20,
                 TR=TR_DEFAULT,
                 skip_processing=False,
                 skip_figures=False,
                 save_slim=False,
                 skip_existing=True):
    """
    Run the full DySCo pipeline end-to-end.

    Parameters
    ----------
    cohort_config : dict
        One entry per participant.  Each value is a dict with keys:
          "cartoon" → folder path (str) OR explicit file list (list of str)
          "rest"    → folder path (str) OR explicit file list (omit if no rest)

        Examples
        --------
        # Folder-based (processes every .nii / .nii.gz / .npy in the folder):
        "P001": {
            "cartoon": r"C:\\data\\P001\\cartoon_runs",
            "rest":    r"C:\\data\\P001\\rest_runs",
        }

        # File-list (explicit files):
        "P001": {
            "cartoon": [r"C:\\data\\P001\\p001_c003.nii",
                        r"C:\\data\\P001\\p001_c005.nii"],
            "rest":    [r"C:\\data\\P001\\p001_r004.nii"],
        }

    output_root       : str   — root directory; .npy files and figures saved here
    half_window_size  : int   — DySCo half-window (full window = 2*h+1).  Default 10
    n_eigen           : int   — number of eigenvectors to retain.  Default 10
    lag               : int   — volume lag for reconfiguration speed.  Default 20
    TR                : float — repetition time in seconds.  Default 2.16
    skip_processing   : bool  — if True, skip NIfTI → .npy and load existing .npy
                                files from <output_root>/<PID>_dysco_output/
    skip_figures      : bool  — if True, skip all figure generation (process only)
    """

    fig_dir = os.path.join(output_root, "pipeline_figures")
    os.makedirs(fig_dir, exist_ok=True)

    participants    = list(cohort_config.keys())
    n_participants  = len(participants)

    # ────────────────────────────────────────────────────────────────────────
    # STEP 1 — Process NIfTI / .npy files  →  DySCo .npy dictionaries
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  STEP 1 / 3 — DySCo Processing  ({n_participants} participants)")
    print(f"{'='*62}")

    all_run_data_by_pid = {}   # {pid: {"cartoon": [(run_id, measures), ...], "rest": [...]}}

    for p_idx, pid in enumerate(participants, 1):
        # Overall participant-level progress header
        print(f"\n  [{p_idx}/{n_participants}]  {pid}")
        _ascii_bar(p_idx - 1, n_participants, label=f"starting {pid}")

        cond_paths = cohort_config[pid]
        pid_out    = os.path.join(output_root, f"{pid}_dysco_output")
        os.makedirs(pid_out, exist_ok=True)
        all_run_data_by_pid[pid] = {"cartoon": [], "rest": []}

        for cond in ["cartoon", "rest"]:
            if cond not in cond_paths:
                continue

            src = cond_paths[cond]
            print(f"\n    Condition: {cond}")

            if skip_processing:
                # ── load pre-computed .npy files ──────────────────────────
                # Naming convention: files with '_c' in name → cartoon,
                # files with '_r' → rest.  Adjust if your naming differs.
                npy_files = sorted(glob.glob(os.path.join(pid_out, "*.npy")))
                for np_path in npy_files:
                    name = os.path.basename(np_path)
                    tag  = "_c" if cond == "cartoon" else "_r"
                    if tag in name:
                        m = np.load(np_path, allow_pickle=True).item()
                        all_run_data_by_pid[pid][cond].append((name, m))
                n_loaded = len(all_run_data_by_pid[pid][cond])
                print(f"    Loaded {n_loaded} pre-computed .npy file(s) for {cond}")

            else:
                # ── run DySCo processing ──────────────────────────────────
                results = batch_process_participant(
                    pid, src, pid_out,
                    half_window_size=half_window_size,
                    n_eigen=n_eigen,
                    lag=lag,
                    show_progress=True,
                    save_slim=save_slim,
                    skip_existing=skip_existing,
                )
                all_run_data_by_pid[pid][cond].extend(results)

        _ascii_bar(p_idx, n_participants, label=f"{pid} done")

    print(f"\n\n{'='*62}")
    print("  STEP 1 complete.")
    print(f"{'='*62}")

    if skip_figures:
        print("\n  skip_figures=True — stopping after processing.")
        return all_run_data_by_pid

    # ────────────────────────────────────────────────────────────────────────
    # STEP 2 — Per-patient figures
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  STEP 2 / 3 — Per-Patient Figures  ({n_participants} participants)")
    print(f"{'='*62}")

    for p_idx, (pid, cond_dict) in enumerate(_make_progress(
            all_run_data_by_pid.items(),
            total=n_participants,
            desc="  per-patient figures"), 1):

        all_runs = cond_dict["cartoon"] + cond_dict["rest"]
        if all_runs:
            print(f"\n  {pid} — three-panel time-course")
            plot_patient_timecourse(pid, all_runs, fig_dir, TR=TR)
        else:
            print(f"  {pid} — no runs found, skipping time-course plot")

    # ────────────────────────────────────────────────────────────────────────
    # STEP 3 — Group figures
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  STEP 3 / 3 — Group Figures")
    print(f"{'='*62}\n")

    group_figures = [
        ("Group average cartoon + Shamshiri block overlay",
         lambda: plot_group_cartoon(all_run_data_by_pid, fig_dir, TR=TR)),
        ("Group average resting state",
         lambda: plot_group_rest(all_run_data_by_pid, fig_dir, TR=TR)),
        ("Waiting period onset-aligned",
         lambda: plot_group_wait_aligned(all_run_data_by_pid, fig_dir, TR=TR)),
        ("Three-condition boxplot (Video / Wait / Rest)",
         lambda: plot_three_condition_boxplot(all_run_data_by_pid, fig_dir)),
        ("Group metric summary boxplot (norm2 / entropy / speed)",
         lambda: plot_group_metric_summary(all_run_data_by_pid, fig_dir)),
    ]

    for g_idx, (label, fn) in enumerate(
            _make_progress(group_figures, total=len(group_figures),
                           desc="  group figures"), 1):
        print(f"\n  [{g_idx}/{len(group_figures)}] {label}")
        try:
            fn()
        except Exception as exc:
            print(f"    WARNING: {label} failed — {exc}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  Pipeline complete.")
    print(f"  Figures saved to:  {fig_dir}")
    print(f"{'='*62}\n")

    return all_run_data_by_pid


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — CONFIGURATION  (edit and uncomment to run)
# ══════════════════════════════════════════════════════════════════════════════
#
#  HOW TO CUSTOMISE
#  ─────────────────
#  1. Set OUTPUT_ROOT to wherever you want .npy files and figures saved.
#  2. Set TR to your scanner's repetition time (seconds).
#  3. Adjust DySCo parameters if needed (HALF_WINDOW, N_EIGEN, LAG).
#  4. Fill in COHORT_CONFIG — one entry per participant.
#     Each participant needs "cartoon" and/or "rest" keys pointing to either:
#       • A folder path  →  every .nii / .nii.gz / .npy in the folder is used.
#       • A list of file paths  →  only those exact files are used.
#  5. Set SKIP_PROCESSING = True to skip the heavy NIfTI → .npy step and just
#     regenerate figures from already-computed .npy files.
#  6. Uncomment the run_pipeline(...) call at the bottom of this block.
#
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── OUTPUT ────────────────────────────────────────────────────────────────
    # ↓ CHANGE: root folder where all outputs are written
    OUTPUT_ROOT = r"C:\Users\naima\DySCo-main\DySCo-main\DELFT_NEW"

    # ── SCANNER PARAMETERS ────────────────────────────────────────────────────
    # ↓ CHANGE: repetition time of your fMRI acquisition (seconds)
    TR = 2.16

    # ── DySCo PARAMETERS ─────────────────────────────────────────────────────
    # ↓ CHANGE (optional): adjust if you want different window / eigen settings
    HALF_WINDOW = 10    # sliding-window half-width  (full window = 2*HALF_WINDOW + 1)
    N_EIGEN     = 10    # number of eigenvectors to retain
    LAG         = 20    # volume lag for reconfiguration speed

    # ── SKIP FLAGS ────────────────────────────────────────────────────────────
    # ↓ CHANGE: set True to reload existing .npy files instead of re-processing
    SKIP_PROCESSING = False
    # ↓ CHANGE: set True to run processing only (no figures generated)
    SKIP_FIGURES    = False

    # ── COHORT CONFIG ─────────────────────────────────────────────────────────
    # ↓ CHANGE: add / remove participants and point paths to your data.
    #
    # Option A — folder path (every .nii / .nii.gz / .npy in folder is used):
    #
    #   "P001": {
    #       "cartoon": r"C:\data\P001\cartoon",
    #       "rest":    r"C:\data\P001\rest",
    #   },
    #
    # Option B — explicit file list:
    #
    #   "P001": {
    #       "cartoon": [
    #           r"C:\data\P001\p001_c003.nii",
    #           r"C:\data\P001\p001_c005.nii",
    #       ],
    #       "rest": [
    #           r"C:\data\P001\p001_r004.nii",
    #           r"C:\data\P001\p001_r006.nii",
    #       ],
    #   },
    #
    # Option C — participant has no rest runs (omit the "rest" key entirely):
    #
    #   "P004": {
    #       "cartoon": r"C:\data\P004\cartoon",
    #   },

    COHORT_CONFIG = {
        # ↓ CHANGE: replace with your participant IDs and data paths
        "P001": {
            "cartoon": r"C:\Users\naima\DySCo-main\DySCo-main\DELFT_NEW\P001",
            "rest":    r"C:\Users\naima\DySCo-main\DySCo-main\DELFT_NEW\P001",
        },
        "P002": {
            "cartoon": r"C:\Users\naima\DySCo-main\DySCo-main\DELFT_NEW\P002",
            "rest":    r"C:\Users\naima\DySCo-main\DySCo-main\DELFT_NEW\P002",
        },
        "P003": {
            "cartoon": r"C:\Users\naima\DySCo-main\DySCo-main\DELFT_NEW\P003",
            "rest":    r"C:\Users\naima\DySCo-main\DySCo-main\DELFT_NEW\P003",
        },
        "P004": {
            "cartoon": r"C:\Users\naima\DySCo-main\DySCo-main\DELFT_NEW\P004",
            "rest":    r"C:\Users\naima\DySCo-main\DySCo-main\DELFT_NEW\P004",
        },
        "P005": {
            "cartoon": r"C:\Users\naima\DySCo-main\DySCo-main\DELFT_NEW\P005",
            "rest":    r"C:\Users\naima\DySCo-main\DySCo-main\DELFT_NEW\P005",
        },
    }

    # ── RUN ───────────────────────────────────────────────────────────────────
    # ↓ Uncomment the block below to run the pipeline.
    # (It is commented out by default so importing this file does not trigger
    #  processing.  Fill in CONFIG above, then remove the comment markers.)

    # run_pipeline(
    #     cohort_config    = COHORT_CONFIG,
    #     output_root      = OUTPUT_ROOT,
    #     half_window_size = HALF_WINDOW,
    #     n_eigen          = N_EIGEN,
    #     lag              = LAG,
    #     TR               = TR,
    #     skip_processing  = SKIP_PROCESSING,
    #     skip_figures     = SKIP_FIGURES,
    # )
