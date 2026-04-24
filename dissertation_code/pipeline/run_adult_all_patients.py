"""
run_adult_all_patients.py
=========================
Runs the full DySCo NIfTI pipeline on all adult participants across ALL
available sessions, then produces:
  - Per-participant concatenated timecourse plots (edge-trimmed, styled)
      Runs ordered: ses-01 c01, r01, c02, r02 → ses-02 c01, r01, c02, r02
  - Per-participant condition stats CSV
  - Group average concatenated plot (ses-01 only, all participants)

Output structure under OUTPUT_ROOT:
  {pid}_dysco_output/
      *_dysco.npy                          <- DySCo measures per run
      concatenated_runs_raw_with_stats/    <- per-patient plots + CSV
  pipeline_figures/                        <- pipeline timecourse / boxplot figs
  group_average/                           <- group average concatenated plots

HOW TO USE
----------
1. Confirm BASE_DIR and OUTPUT_ROOT below are correct.
2. Set SKIP_PROCESSING = True after the pipeline has run once (skips heavy
   NIfTI → .npy step, just regenerates plots).
3. Run:  python run_adult_all_patients.py
"""

import os
import sys
import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import uniform_filter1d

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False

try:
    from scipy.stats import ttest_ind
    _SCIPY = True
except ImportError:
    _SCIPY = False

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from dysco_nifti_pipeline import (run_pipeline,
                                   plot_group_cartoon,
                                   plot_group_rest,
                                   plot_group_wait_aligned)

# ==============================================================================
# CONFIG — edit these two paths, everything else is automatic
# ==============================================================================

BASE_DIR    = r"D:\encrypt_generalised_adult\ADULT"
OUTPUT_ROOT = r"D:\encrypt_generalised_adult\ADULT\dysco_results"

# Set True after first run to skip NIfTI processing and just redo plots
SKIP_PROCESSING = False

# ==============================================================================
# Participants already processed — skip pipeline but include in group average.
# Maps pid → folder containing their *_dysco.npy files.
# ==============================================================================

ALREADY_PROCESSED = {
    "sub-ga01": r"D:\encrypt_generalised_adult\ADULT\dysco_results\sub-ga01_dysco_output",
    "sub-ga02": r"D:\encrypt_generalised_adult\ADULT\dysco_results\sub-ga02_dysco_output",
    "sub-ga03": r"D:\encrypt_generalised_adult\ADULT\dysco_results\sub-ga03_dysco_output",
    "sub-ga04": r"D:\encrypt_generalised_adult\ADULT\dysco_results\sub-ga04_dysco_output",
    "sub-ga05": r"D:\encrypt_generalised_adult\ADULT\dysco_results\sub-ga05_dysco_output",
    "sub-ga08": r"D:\encrypt_generalised_adult\ADULT\dysco_results\sub-ga08_dysco_output",
    "sub-ga09": r"D:\encrypt_generalised_adult\ADULT\dysco_results\sub-ga09_dysco_output",
    "sub-ga11": r"D:\encrypt_generalised_adult\ADULT\dysco_results\sub-ga11_dysco_output",
    "sub-ga12": r"D:\encrypt_generalised_adult\ADULT\dysco_results\sub-ga12_dysco_output",
    "sub-ga15": r"D:\encrypt_generalised_adult\ADULT\dysco_results\sub-ga15_dysco_output",
}

# ==============================================================================
# Pipeline / plot parameters
# ==============================================================================

TR          = 2.16
HALF_WINDOW = 10
N_EIGEN     = 10
LAG         = 20
SMOOTH_WIN  = 5
EDGE_TRIM   = 10   # windows clipped from each run's start/end (removes edge artefacts)

C_CARTOON = "#4472C4"
C_REST    = "#C0504D"

METRIC_YLABELS = {
    "entropy": "DySCo Entropy",
    "speed":   "Reconfiguration Speed",
    "norm2":   "Connectivity Norm (L2)",
}

# ── file detection ─────────────────────────────────────────────────────────────

def _find_participant_runs(base_dir, pid):
    """
    Return (cartoon_files, rest_files, run_labels) across all sessions.

    Prefers rasub-* (realigned + slice-time corrected) over raw sub-* files.
    Searches in:  ses-01/func,  ses-02/func,  func   (in that order)
    Runs are ordered ses-01 first, then ses-02, within each session: c01 r01 c02 r02.
    """
    pid_short = pid.replace("sub-", "")

    cartoon_files, rest_files, run_labels = [], [], []

    for ses in ["ses-01", "ses-02", None]:   # None → bare func/ folder
        if ses:
            fd = os.path.join(base_dir, pid, ses, "func")
            ses_tag = ses.replace("ses-", "s")   # "s1", "s2"
        else:
            fd = os.path.join(base_dir, pid, "func")
            ses_tag = "s1"

        if not os.path.exists(fd):
            continue

        # prefer rasub- prefix, fall back to sub-
        ses_label = ses if ses else "ses-01"
        for prefix in [f"rasub-{pid_short}_{ses_label}",
                       f"{pid}_{ses_label}"]:
            found = {}
            for run_n in [1, 2]:
                c = os.path.join(fd, f"{prefix}_task-cartoon_run-0{run_n}_bold.nii")
                r = os.path.join(fd, f"{prefix}_task-rest_run-0{run_n}_bold.nii")
                if os.path.exists(c) and os.path.exists(r):
                    found[run_n] = (c, r)

            if found:
                # add runs in order: c01 r01 c02 r02
                for run_n in sorted(found):
                    c, r = found[run_n]
                    cartoon_files.append(c)
                    rest_files.append(r)
                    # labels for concatenated plot
                    run_labels.append((f"{ses_tag}c0{run_n}", f"{ses_tag}r0{run_n}"))
                break   # stop trying prefixes once found for this session

    return cartoon_files, rest_files, run_labels


def _smooth(sig, w):
    if not w or w <= 1 or len(sig) < w:
        return sig.copy()
    # mode='nearest' repeats edge values instead of zero-padding,
    # preventing the sharp drops at run boundaries that mode='same' causes
    return uniform_filter1d(sig.astype(float), size=w, mode='nearest')


# ── concatenated analysis helpers ─────────────────────────────────────────────

def _build_records(dysco_folder, run_labels, pid):
    """
    Load .npy files in the correct run order and tag with run_type + label.
    Matches files using run order info from run_labels (list of (c_label, r_label) tuples).
    """
    all_npy = glob.glob(os.path.join(dysco_folder, "*_dysco.npy"))

    def _stem_to_key(fname):
        """Extract session + task + run identifier from filename."""
        m = re.search(r"(ses-\d+).*task-(cartoon|rest).*run-(\d+)", fname)
        if m:
            return m.group(1), m.group(2), m.group(3)
        return None

    # Index npy files by (session, task, run)
    npy_index = {}
    for fpath in all_npy:
        key = _stem_to_key(os.path.basename(fpath))
        if key:
            npy_index[key] = fpath

    # Also handle files without ses- tag (ga02/ga03 style)
    for fpath in all_npy:
        fname = os.path.basename(fpath)
        m = re.search(r"task-(cartoon|rest).*run-(\d+)", fname)
        if m and not re.search(r"ses-\d+", fname):
            npy_index[("ses-01", m.group(1), m.group(2))] = fpath

    records = []
    seen_sessions = []
    for ses in ["ses-01", "ses-02"]:
        for run_n in ["01", "02"]:
            for task, rt in [("cartoon", "c"), ("rest", "r")]:
                key = (ses, task, run_n)
                if key not in npy_index:
                    continue
                try:
                    data = np.load(npy_index[key], allow_pickle=True).item()
                except Exception as e:
                    print(f"  [{pid}] Could not load {os.path.basename(npy_index[key])}: {e}")
                    continue
                # build a short label like "s1c01"
                ses_tag = ses.replace("ses-", "s")
                label   = f"{ses_tag}{rt}0{run_n[-1]}"
                records.append({
                    "run_type": rt,
                    "label":    label,
                    "session":  ses,
                    "data":     data,
                })

    return records


def _plot_concatenated(records, measure, ylabel, out_dir, pshort,
                       smooth_window=None, title_prefix=""):
    """Concatenated timecourse with edge trimming and blue/red condition styling."""
    boundaries, pos = [], 0
    for rec in records:
        if measure not in rec["data"]:
            continue
        arr     = np.asarray(rec["data"][measure], dtype=float)
        trimmed = arr[EDGE_TRIM: len(arr) - EDGE_TRIM]
        if len(trimmed) == 0:
            continue
        boundaries.append((pos, pos + len(trimmed), rec["run_type"], rec["label"]))
        pos += len(trimmed)

    if not boundaries:
        return

    tag    = "_raw" if smooth_window is None else f"_smoothed_w{smooth_window}"
    suffix = "raw"  if smooth_window is None else f"smoothed w={smooth_window}"

    fig, ax = plt.subplots(figsize=(18, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # add session divider lines
    session_boundaries = set()
    prev_ses = None
    bpos = 0
    for rec in records:
        if measure not in rec["data"]:
            continue
        arr = np.asarray(rec["data"][measure], dtype=float)
        n   = len(arr[EDGE_TRIM: len(arr) - EDGE_TRIM])
        if n == 0:
            continue
        if prev_ses and rec["session"] != prev_ses:
            session_boundaries.add(bpos)
        prev_ses = rec["session"]
        bpos += n

    for s, e, rt, lbl in boundaries:
        ax.axvspan(s, e, color="#DDEEFF" if rt == "c" else "#FFDDDD",
                   alpha=0.55, zorder=0)
        ax.axvline(s, color="#aaaaaa", linestyle="--", linewidth=0.8,
                   alpha=0.7, zorder=1)
        ax.text((s + e) / 2, 0.985, lbl,
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, color="#333333")

    for xpos in session_boundaries:
        ax.axvline(xpos, color="#333333", linestyle="-", linewidth=1.5,
                   alpha=0.6, zorder=2)

    if boundaries:
        ax.axvline(boundaries[-1][1], color="#aaaaaa", linestyle="--",
                   linewidth=0.8, alpha=0.7)

    pos = 0
    for rec in records:
        if measure not in rec["data"]:
            continue
        arr     = np.asarray(rec["data"][measure], dtype=float)
        trimmed = arr[EDGE_TRIM: len(arr) - EDGE_TRIM]
        if len(trimmed) == 0:
            continue
        seg = _smooth(trimmed, smooth_window)
        ax.plot(range(pos, pos + len(seg)), seg,
                color=C_CARTOON if rec["run_type"] == "c" else C_REST,
                linewidth=1.3, zorder=3)
        pos += len(trimmed)

    ax.set_title(f"{title_prefix}  -  DySCo {ylabel} across consecutive runs ({suffix})",
                 fontsize=13, pad=10)
    ax.set_xlabel("Time windows across consecutive runs", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles=[mpatches.Patch(facecolor=C_CARTOON, label="Video (cartoon)"),
                 mpatches.Patch(facecolor=C_REST,    label="Rest ('please wait')")],
        loc="upper right", fontsize=9, framealpha=0.9, edgecolor="#cccccc"
    )
    plt.tight_layout()
    path = os.path.join(out_dir, f"{pshort}_{measure}_concatenated{tag}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {os.path.basename(path)}")


def _condition_stats(records, pid, pshort, out_dir):
    """Cartoon vs rest stats CSV across all sessions."""
    rows = []
    for measure in ["entropy", "speed", "norm1", "norm2"]:
        c_vals = np.concatenate([
            np.asarray(r["data"][measure], dtype=float)
            for r in records
            if r["run_type"] == "c" and measure in r["data"]
        ]) if any(r["run_type"] == "c" and measure in r["data"] for r in records) \
          else np.array([])

        r_vals = np.concatenate([
            np.asarray(r["data"][measure], dtype=float)
            for r in records
            if r["run_type"] == "r" and measure in r["data"]
        ]) if any(r["run_type"] == "r" and measure in r["data"] for r in records) \
          else np.array([])

        row = {
            "patient_id":   pid,
            "measure":      measure,
            "cartoon_n":    len(c_vals),
            "rest_n":       len(r_vals),
            "cartoon_mean": np.mean(c_vals) if len(c_vals) else float("nan"),
            "cartoon_std":  np.std(c_vals)  if len(c_vals) else float("nan"),
            "rest_mean":    np.mean(r_vals) if len(r_vals) else float("nan"),
            "rest_std":     np.std(r_vals)  if len(r_vals) else float("nan"),
            "mean_diff_cartoon_minus_rest": (
                np.mean(c_vals) - np.mean(r_vals)
                if len(c_vals) and len(r_vals) else float("nan")
            ),
            "t_statistic": float("nan"),
            "p_value":     float("nan"),
        }
        if _SCIPY and len(c_vals) > 1 and len(r_vals) > 1:
            t, p = ttest_ind(c_vals, r_vals, equal_var=False)
            row["t_statistic"] = t
            row["p_value"]     = p
        rows.append(row)

    if _PANDAS:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(out_dir, f"{pshort}_condition_stats.csv")
        df.to_csv(csv_path, index=False)
        print(f"    Saved: {os.path.basename(csv_path)}")


def _run_concatenated_analysis(pid, pshort, run_labels, out_root,
                               npy_folder=None):
    """
    Load .npy files, plot concatenated timecourses, save stats CSV.

    npy_folder : override where .npy files are loaded from (for already-processed
                 participants whose files live outside OUTPUT_ROOT).
                 Plots are always saved under out_root/{pid}_dysco_output/.
    """
    load_from  = npy_folder or os.path.join(out_root, f"{pid}_dysco_output")
    concat_dir = os.path.join(out_root, f"{pid}_dysco_output",
                              "concatenated_runs_raw_with_stats")
    os.makedirs(concat_dir, exist_ok=True)

    records = _build_records(load_from, run_labels, pid)
    if not records:
        print(f"  [{pid}] No .npy files found in {load_from} — skipping.")
        return []

    n_ses = len(set(r["session"] for r in records))
    print(f"  [{pid}] {len(records)} runs across {n_ses} session(s) — generating plots...")

    for measure, ylabel in METRIC_YLABELS.items():
        _plot_concatenated(records, measure, ylabel, concat_dir, pshort,
                           smooth_window=None,       title_prefix=pid)
        _plot_concatenated(records, measure, ylabel, concat_dir, pshort,
                           smooth_window=SMOOTH_WIN, title_prefix=pid)

    _condition_stats(records, pid, pshort, concat_dir)
    return records


# ── group average ──────────────────────────────────────────────────────────────

def _get_ses1_signal(records, measure):
    """Return edge-trimmed concatenated signal for ses-01 runs only."""
    parts = []
    for rec in records:
        if rec["session"] != "ses-01" or measure not in rec["data"]:
            continue
        arr     = np.asarray(rec["data"][measure], dtype=float)
        trimmed = arr[EDGE_TRIM: len(arr) - EDGE_TRIM]
        if len(trimmed) > 0:
            parts.append(trimmed)
    return np.concatenate(parts) if parts else np.array([])


def _plot_group_average(all_records_by_pid, out_dir):
    """
    Group average: single stacked figure with one panel per metric (entropy,
    speed, norm2), all sharing the ses-01 c01→r01→c02→r02 x-axis structure.
    Mean ± SD shaded band, blue/red condition colouring.
    Saved as raw and smoothed versions.
    """
    os.makedirs(out_dir, exist_ok=True)
    pids = list(all_records_by_pid.keys())
    n    = len(pids)

    # ses-01 run records from the first participant (used for boundaries)
    first_ses1 = [r for r in next(iter(all_records_by_pid.values()))
                  if r["session"] == "ses-01"]

    # Build x-axis boundary positions (use entropy as reference length)
    boundaries, bpos = [], 0
    for rec in first_ses1:
        if "entropy" not in rec["data"]:
            continue
        arr     = np.asarray(rec["data"]["entropy"], dtype=float)
        trimmed = arr[EDGE_TRIM: len(arr) - EDGE_TRIM]
        if len(trimmed) == 0:
            continue
        boundaries.append((bpos, bpos + len(trimmed), rec["run_type"], rec["label"]))
        bpos += len(trimmed)

    # Pre-compute group mean ± SD per metric using NaN-padding so that
    # participants missing a run (e.g. ga04 ses-01_r02) don't truncate everyone.
    metric_stats = {}
    for measure in METRIC_YLABELS:
        signals = []
        for pid in pids:
            sig = _get_ses1_signal(all_records_by_pid[pid], measure)
            if len(sig) > 0:
                signals.append(sig)
        if not signals:
            continue
        max_len = max(len(s) for s in signals)
        # Pad shorter signals with NaN so every row is max_len
        mat = np.full((len(signals), max_len), np.nan)
        for i, s in enumerate(signals):
            mat[i, :len(s)] = s
        metric_stats[measure] = {
            "mean":    np.nanmean(mat, axis=0),
            "sd":      np.nanstd(mat,  axis=0),
            "max_len": max_len,
            "n":       len(signals),
        }

    n_panels = len(metric_stats)
    if n_panels == 0:
        print("  No data for group average.")
        return

    for tag, sw in [("_raw", None), (f"_smoothed_w{SMOOTH_WIN}", SMOOTH_WIN)]:
        suffix = "raw" if sw is None else f"smoothed  w = {sw}"

        fig, axes = plt.subplots(n_panels, 1,
                                 figsize=(18, 4 * n_panels),
                                 sharex=False)
        if n_panels == 1:
            axes = [axes]
        fig.patch.set_facecolor("white")

        for ax, (measure, ylabel) in zip(axes, METRIC_YLABELS.items()):
            if measure not in metric_stats:
                ax.set_visible(False)
                continue

            stats   = metric_stats[measure]
            max_len = stats["max_len"]
            gm      = _smooth(stats["mean"], sw)
            gs      = _smooth(stats["sd"],   sw)

            ax.set_facecolor("white")

            # coloured background + dashed run dividers + run labels
            for s, e, rt, lbl in boundaries:
                if s >= max_len:
                    break
                e_clip = min(e, max_len)
                ax.axvspan(s, e_clip,
                           color="#DDEEFF" if rt == "c" else "#FFDDDD",
                           alpha=0.55, zorder=0)
                ax.axvline(s, color="#aaaaaa", linestyle="--",
                           linewidth=0.8, alpha=0.7, zorder=1)
                ax.text((s + e_clip) / 2, 0.985, lbl,
                        transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=9, color="#333333")
            if boundaries:
                ax.axvline(min(boundaries[-1][1], max_len),
                           color="#aaaaaa", linestyle="--",
                           linewidth=0.8, alpha=0.7, zorder=1)

            # mean ± SD drawn segment-by-segment, coloured by condition
            gpos = 0
            for rec in first_ses1:
                if measure not in rec["data"]:
                    continue
                arr     = np.asarray(rec["data"][measure], dtype=float)
                trimmed = arr[EDGE_TRIM: len(arr) - EDGE_TRIM]
                n_seg   = min(len(trimmed), max_len - gpos)
                if n_seg <= 0:
                    break
                x  = np.arange(gpos, gpos + n_seg)
                lc = C_CARTOON if rec["run_type"] == "c" else C_REST
                # mask NaN positions (windows where some participants have no data)
                valid = ~np.isnan(gm[gpos:gpos + n_seg])
                if valid.any():
                    ax.fill_between(x[valid],
                                    (gm[gpos:gpos + n_seg] - gs[gpos:gpos + n_seg])[valid],
                                    (gm[gpos:gpos + n_seg] + gs[gpos:gpos + n_seg])[valid],
                                    color=lc, alpha=0.20, zorder=1)
                    ax.plot(x[valid], gm[gpos:gpos + n_seg][valid],
                            color=lc, linewidth=1.8, zorder=2)
                gpos += n_seg

            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_xlim(0, max_len)
            ax.grid(True, alpha=0.25, linewidth=0.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(
                handles=[mpatches.Patch(facecolor=C_CARTOON, label="Video (cartoon)"),
                         mpatches.Patch(facecolor=C_REST,    label="Rest ('please wait')")],
                loc="upper right", fontsize=8.5, framealpha=0.9, edgecolor="#cccccc"
            )

        axes[-1].set_xlabel("Time windows across consecutive runs", fontsize=11)
        fig.suptitle(
            f"Group Average DySCo Metrics  —  Session 1  |  "
            f"n = {n} participants  |  mean ± SD  ({suffix})",
            fontsize=13, fontweight="bold", y=1.01
        )
        plt.tight_layout()
        path = os.path.join(out_dir, f"group_all_metrics_concatenated{tag}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {os.path.basename(path)}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # ── detect all participants ───────────────────────────────────────────────
    all_pids = sorted([
        d for d in os.listdir(BASE_DIR)
        if d.startswith("sub-ga") and "dysco" not in d
        and os.path.isdir(os.path.join(BASE_DIR, d))
    ])

    print(f"\n{'='*62}")
    print(f"  Adult DySCo Pipeline — {len(all_pids)} participants")
    print(f"  Output root: {OUTPUT_ROOT}")
    print(f"{'='*62}\n")

    cohort_config  = {}
    run_labels_map = {}
    skipped        = []

    for pid in all_pids:
        if pid in ALREADY_PROCESSED:
            # detect run labels for loading later, but skip from pipeline
            _, _, run_labels = _find_participant_runs(BASE_DIR, pid)
            run_labels_map[pid] = run_labels
            print(f"  {pid}: already processed — will load from existing folder.")
            continue

        cartoon_files, rest_files, run_labels = _find_participant_runs(BASE_DIR, pid)
        if not cartoon_files:
            print(f"  WARNING: no NIfTI files found for {pid} — skipping.")
            skipped.append(pid)
            continue
        cohort_config[pid]  = {"cartoon": cartoon_files, "rest": rest_files}
        run_labels_map[pid] = run_labels
        print(f"  {pid}: {len(cartoon_files)} cartoon + {len(rest_files)} rest runs")

    if skipped:
        print(f"\n  Skipped (missing data): {skipped}")
    if not cohort_config and not ALREADY_PROCESSED:
        print("ERROR: No participants found. Check BASE_DIR.")
        sys.exit(1)

    # ── Step 1: DySCo pipeline (new participants only) ───────────────────────
    print(f"\n{'='*62}")
    print(f"  STEP 1/3 — DySCo pipeline ({len(cohort_config)} new participants)")
    if ALREADY_PROCESSED:
        print(f"  Skipping (already done): {list(ALREADY_PROCESSED.keys())}")
    print(f"{'='*62}\n")

    if cohort_config:
        run_pipeline(
            cohort_config    = cohort_config,
            output_root      = OUTPUT_ROOT,
            half_window_size = HALF_WINDOW,
            n_eigen          = N_EIGEN,
            lag              = LAG,
            TR               = TR,
            skip_processing  = SKIP_PROCESSING,
            skip_figures     = False,
            save_slim        = True,   # skip saving eigenvectors (3GB each)
            skip_existing    = True,   # skip NIfTIs that already have a valid .npy
        )

    # ── Step 2: per-patient concatenated plots ───────────────────────────────
    print(f"\n{'='*62}")
    print("  STEP 2/3 — Per-participant concatenated timecourses")
    print(f"{'='*62}\n")

    all_records_by_pid = {}

    # Load already-processed participants from their existing npy location
    for pid, npy_folder in ALREADY_PROCESSED.items():
        pshort  = pid.replace("sub-", "")
        run_lbl = run_labels_map.get(pid, [])
        print(f"  [{pid}] Loading from existing folder...")
        records = _run_concatenated_analysis(pid, pshort, run_lbl,
                                             OUTPUT_ROOT, npy_folder=npy_folder)
        if records:
            all_records_by_pid[pid] = records

    # New participants
    for pid in cohort_config:
        pshort  = pid.replace("sub-", "")
        records = _run_concatenated_analysis(pid, pshort, run_labels_map[pid],
                                             OUTPUT_ROOT)
        if records:
            all_records_by_pid[pid] = records

    # ── Step 3: concatenated group average (c01→r01→c02→r02) ─────────────────
    print(f"\n{'='*62}")
    print(f"  STEP 3/4 — Concatenated group average (n={len(all_records_by_pid)} participants, ses-01)")
    print(f"{'='*62}\n")

    group_dir = os.path.join(OUTPUT_ROOT, "group_average")
    if all_records_by_pid:
        _plot_group_average(all_records_by_pid, group_dir)
    else:
        print("  No participant data available for group average.")

    # ── Step 4: pipeline-style group figures (cartoon / rest / wait) ──────────
    # Uses the same three-panel style as the paediatric DELFT_NEW figures.
    # Cartoon figure has Shamshiri block overlay; wait figure is onset-aligned.
    # Only ses-01 runs used so every participant contributes equally.
    print(f"\n{'='*62}")
    print(f"  STEP 4/4 — Pipeline-style group figures (cartoon / rest / wait)")
    print(f"{'='*62}\n")

    # Convert records to the format expected by the pipeline figure functions:
    #   { pid: { "cartoon": [(run_id, measures_dict), ...],
    #            "rest":    [(run_id, measures_dict), ...] } }
    pipeline_data = {}
    for pid, records in all_records_by_pid.items():
        cartoon = [(r["label"], r["data"]) for r in records
                   if r["run_type"] == "c" and r["session"] == "ses-01"]
        rest    = [(r["label"], r["data"]) for r in records
                   if r["run_type"] == "r" and r["session"] == "ses-01"]
        if cartoon or rest:
            pipeline_data[pid] = {"cartoon": cartoon, "rest": rest}

    if pipeline_data:
        print(f"  Generating pipeline-style figures for {len(pipeline_data)} participants...")
        try:
            plot_group_cartoon(pipeline_data, group_dir, TR=TR)
        except Exception as e:
            print(f"  WARNING: cartoon figure failed — {e}")
        try:
            plot_group_rest(pipeline_data, group_dir, TR=TR)
        except Exception as e:
            print(f"  WARNING: rest figure failed — {e}")
        try:
            plot_group_wait_aligned(pipeline_data, group_dir, TR=TR)
        except Exception as e:
            print(f"  WARNING: wait-aligned figure failed — {e}")
    else:
        print("  No data for pipeline-style figures.")

    print(f"\n{'='*62}")
    print("  All done.")
    print(f"  Individual outputs : {OUTPUT_ROOT}/<pid>_dysco_output/")
    print(f"  Group average      : {group_dir}")
    print(f"{'='*62}\n")
