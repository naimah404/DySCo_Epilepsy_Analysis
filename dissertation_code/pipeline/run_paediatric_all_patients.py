"""
run_paediatric_all_patients.py
================================
Runs the full DySCo NIfTI pipeline on all paediatric epilepsy participants
(P001–P005, DELFT_NEW dataset), then produces:
  - Per-participant concatenated timecourse plots
  - Per-participant condition stats CSV
  - Group average concatenated plot
  - Pipeline-style group figures (cartoon / rest / wait-aligned)

File naming convention:
  DELFT_NEW/P001/p001_c003_merged.nii   (cartoon)
  DELFT_NEW/P001/p001_r004_merged.nii   (rest)
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
# CONFIG
# ==============================================================================

BASE_DIR    = r"C:\Users\naima\DySCo-main\DySCo-main\DELFT_NEW"
OUTPUT_ROOT = BASE_DIR   # .npy output folders (P001_dysco_output etc.) sit here

SKIP_PROCESSING = True   # .npy files already exist — set False to reprocess

# Run IDs per participant — maps to NIfTI files p00X_<run_id>_merged.nii
RUNS = {
    "P001": {"cartoon": ["c003", "c005"], "rest": ["r004", "r006"]},
    "P002": {"cartoon": ["c003", "c005"], "rest": ["r004", "r006"]},
    "P003": {"cartoon": ["c003", "c006"], "rest": ["r004", "r007"]},
    "P004": {"cartoon": ["c004", "c005"], "rest": ["r003"]},
    "P005": {"cartoon": ["c003", "c006"], "rest": ["r004"]},
}

# ==============================================================================
# Pipeline / plot parameters
# ==============================================================================

TR          = 2.16
HALF_WINDOW = 10
N_EIGEN     = 10
LAG         = 20
SMOOTH_WIN  = 5
EDGE_TRIM   = 10

C_CARTOON = "#4472C4"
C_REST    = "#C0504D"

METRIC_YLABELS = {
    "entropy": "DySCo Entropy",
    "speed":   "Reconfiguration Speed",
    "norm2":   "Connectivity Norm (L2)",
}

# ── file detection ─────────────────────────────────────────────────────────────

def _build_cohort_config():
    """Build cohort_config dict mapping pid → {cartoon: [paths], rest: [paths]}."""
    config = {}
    for pid, run_info in RUNS.items():
        pnum = pid.lower()
        pid_dir = os.path.join(BASE_DIR, pid)
        cartoon_files, rest_files = [], []

        for run_id in run_info["cartoon"]:
            f = os.path.join(pid_dir, f"{pnum}_{run_id}_merged.nii")
            if os.path.exists(f):
                cartoon_files.append(f)
            else:
                print(f"  WARNING: missing {f}")

        for run_id in run_info["rest"]:
            f = os.path.join(pid_dir, f"{pnum}_{run_id}_merged.nii")
            if os.path.exists(f):
                rest_files.append(f)
            else:
                print(f"  WARNING: missing {f}")

        if cartoon_files or rest_files:
            config[pid] = {"cartoon": cartoon_files, "rest": rest_files}
            print(f"  {pid}: {len(cartoon_files)} cartoon + {len(rest_files)} rest runs")

    return config


def _smooth(sig, w):
    if not w or w <= 1 or len(sig) < w:
        return sig.copy()
    return uniform_filter1d(sig.astype(float), size=w, mode='nearest')


# ── concatenated analysis ─────────────────────────────────────────────────────

def _build_records(pid):
    """Load all .npy files for a participant in run order."""
    pnum   = pid.lower()
    folder = os.path.join(OUTPUT_ROOT, f"{pid}_dysco_output")
    records = []

    for run_id in RUNS[pid]["cartoon"]:
        fpath = os.path.join(folder, f"{pnum}_{run_id}_merged_dysco.npy")
        if not os.path.exists(fpath):
            print(f"  [{pid}] Missing: {os.path.basename(fpath)}")
            continue
        try:
            data = np.load(fpath, allow_pickle=True).item()
        except Exception as e:
            print(f"  [{pid}] Could not load {os.path.basename(fpath)}: {e}")
            continue
        records.append({"run_type": "c", "label": run_id, "data": data})

    for run_id in RUNS[pid]["rest"]:
        fpath = os.path.join(folder, f"{pnum}_{run_id}_merged_dysco.npy")
        if not os.path.exists(fpath):
            print(f"  [{pid}] Missing: {os.path.basename(fpath)}")
            continue
        try:
            data = np.load(fpath, allow_pickle=True).item()
        except Exception as e:
            print(f"  [{pid}] Could not load {os.path.basename(fpath)}: {e}")
            continue
        records.append({"run_type": "r", "label": run_id, "data": data})

    return records


def _plot_concatenated(records, measure, ylabel, out_dir, pid,
                       smooth_window=None):
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

    for s, e, rt, lbl in boundaries:
        ax.axvspan(s, e, color="#DDEEFF" if rt == "c" else "#FFDDDD",
                   alpha=0.55, zorder=0)
        ax.axvline(s, color="#aaaaaa", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.text((s + e) / 2, 0.985, lbl,
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, color="#333333")
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

    ax.set_title(f"{pid}  —  DySCo {ylabel} ({suffix})", fontsize=13, pad=10)
    ax.set_xlabel("Time windows across consecutive runs", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles=[mpatches.Patch(facecolor=C_CARTOON, label="Cartoon"),
                 mpatches.Patch(facecolor=C_REST,    label="Rest")],
        loc="upper right", fontsize=9, framealpha=0.9
    )
    plt.tight_layout()
    path = os.path.join(out_dir, f"{pid.lower()}_{measure}_concatenated{tag}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {os.path.basename(path)}")


def _condition_stats(records, pid, out_dir):
    rows = []
    for measure in ["entropy", "speed", "norm2"]:
        c_vals = np.concatenate([
            np.asarray(r["data"][measure], dtype=float)
            for r in records if r["run_type"] == "c" and measure in r["data"]
        ]) if any(r["run_type"] == "c" and measure in r["data"] for r in records) \
          else np.array([])
        r_vals = np.concatenate([
            np.asarray(r["data"][measure], dtype=float)
            for r in records if r["run_type"] == "r" and measure in r["data"]
        ]) if any(r["run_type"] == "r" and measure in r["data"] for r in records) \
          else np.array([])

        row = {
            "patient_id":   pid,
            "measure":      measure,
            "cartoon_mean": np.mean(c_vals) if len(c_vals) else float("nan"),
            "rest_mean":    np.mean(r_vals) if len(r_vals) else float("nan"),
            "mean_diff":    np.mean(c_vals) - np.mean(r_vals)
                            if len(c_vals) and len(r_vals) else float("nan"),
        }
        if _SCIPY and len(c_vals) > 1 and len(r_vals) > 1:
            from scipy.stats import ttest_ind
            t, p = ttest_ind(c_vals, r_vals, equal_var=False)
            row["t_statistic"] = t
            row["p_value"]     = p
        rows.append(row)

    if _PANDAS:
        df = pd.DataFrame(rows)
        csv_path = os.path.join(out_dir, f"{pid.lower()}_condition_stats.csv")
        df.to_csv(csv_path, index=False)
        print(f"    Saved: {os.path.basename(csv_path)}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print(f"\n{'='*62}")
    print(f"  Paediatric DySCo Pipeline — {len(RUNS)} participants")
    print(f"  Output root: {OUTPUT_ROOT}")
    print(f"{'='*62}\n")

    cohort_config = _build_cohort_config()

    # ── Step 1: DySCo pipeline ────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  STEP 1/3 — DySCo pipeline (SKIP_PROCESSING={SKIP_PROCESSING})")
    print(f"{'='*62}\n")

    if not SKIP_PROCESSING:
        run_pipeline(
            cohort_config    = cohort_config,
            output_root      = OUTPUT_ROOT,
            half_window_size = HALF_WINDOW,
            n_eigen          = N_EIGEN,
            lag              = LAG,
            TR               = TR,
            skip_processing  = False,
            skip_figures     = False,
            save_slim        = True,
            skip_existing    = True,
        )

    # ── Step 2: per-patient concatenated plots ────────────────────────────────
    print(f"\n{'='*62}")
    print("  STEP 2/3 — Per-participant concatenated timecourses")
    print(f"{'='*62}\n")

    all_records_by_pid = {}

    for pid in RUNS:
        concat_dir = os.path.join(OUTPUT_ROOT, f"{pid}_dysco_output",
                                  "concatenated_runs_raw_with_stats")
        os.makedirs(concat_dir, exist_ok=True)

        records = _build_records(pid)
        if not records:
            print(f"  [{pid}] No .npy files found — skipping.")
            continue
        print(f"  [{pid}] {len(records)} runs — generating plots...")

        for measure, ylabel in METRIC_YLABELS.items():
            _plot_concatenated(records, measure, ylabel, concat_dir, pid,
                               smooth_window=None)
            _plot_concatenated(records, measure, ylabel, concat_dir, pid,
                               smooth_window=SMOOTH_WIN)
        _condition_stats(records, pid, concat_dir)
        all_records_by_pid[pid] = records

    # ── Step 3: pipeline-style group figures ─────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  STEP 3/3 — Pipeline-style group figures (cartoon / rest / wait)")
    print(f"{'='*62}\n")

    group_dir = os.path.join(OUTPUT_ROOT, "group_average")
    os.makedirs(group_dir, exist_ok=True)

    pipeline_data = {}
    for pid, records in all_records_by_pid.items():
        cartoon = [(r["label"], r["data"]) for r in records if r["run_type"] == "c"]
        rest    = [(r["label"], r["data"]) for r in records if r["run_type"] == "r"]
        if cartoon or rest:
            pipeline_data[pid] = {"cartoon": cartoon, "rest": rest}

    if pipeline_data:
        for fn, label in [(plot_group_cartoon,     "cartoon"),
                          (plot_group_rest,         "rest"),
                          (plot_group_wait_aligned, "wait-aligned")]:
            try:
                fn(pipeline_data, group_dir, TR=TR)
            except Exception as e:
                print(f"  WARNING: {label} figure failed — {e}")

    print(f"\n{'='*62}")
    print("  All done.")
    print(f"  Individual outputs : {OUTPUT_ROOT}/<PID>_dysco_output/")
    print(f"  Group average      : {group_dir}")
    print(f"{'='*62}\n")
