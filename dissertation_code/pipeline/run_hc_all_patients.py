"""
run_hc_all_patients.py
======================
Runs the full DySCo NIfTI pipeline on all healthy-control (HC) participants
at D:/adult_controls/Adult/, then produces:
  - Per-participant concatenated timecourse plots
  - Per-participant condition stats CSV
  - Group average concatenated plot (ses-01 only)
  - Pipeline-style group figures (cartoon / rest / wait-aligned)

File naming convention for HC data:
  asub-hcXX_ses-01_task-cartoon_run-01_bold.nii
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

BASE_DIR    = r"D:\adult_controls\Adult"
OUTPUT_ROOT = r"D:\adult_controls\Adult\dysco_results"


SKIP_PROCESSING = False

PREPROCESSED_PARTICIPANTS = {"sub-hc01", "sub-hc02", "sub-hc03", "sub-hc04"}

ALREADY_PROCESSED = {
    pid: os.path.join(OUTPUT_ROOT, f"{pid}_dysco_output")
    for pid in PREPROCESSED_PARTICIPANTS
}
# ==============================================================================
# Pipeline / plot parameters  (identical to adult epilepsy pipeline)
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

def _find_participant_runs(base_dir, pid):
    """
    Return (cartoon_files, rest_files, run_labels) for ses-01.
    HC files are named:  asub-hcXX_ses-01_task-cartoon_run-01_bold.nii
    """
    pid_short = pid.replace("sub-", "")   # e.g. "hc01"
    cartoon_files, rest_files, run_labels = [], [], []

    for ses in ["ses-01", "ses-02", None]:
        ses_label = ses if ses else "ses-01"
        ses_tag   = ses.replace("ses-", "s") if ses else "s1"
        fd = os.path.join(base_dir, pid, ses_label, "func") if ses \
             else os.path.join(base_dir, pid, "func")
        if not os.path.exists(fd):
            continue

        # HC prefix is "asub-hcXX_ses-XX"; also try without leading 'a' as fallback
        for prefix in [f"asub-{pid_short}_{ses_label}",
                       f"sub-{pid_short}_{ses_label}",
                       f"{pid}_{ses_label}"]:
            found = {}
            for run_n in [1, 2]:
                c = os.path.join(fd, f"{prefix}_task-cartoon_run-0{run_n}_bold.nii")
                r = os.path.join(fd, f"{prefix}_task-rest_run-0{run_n}_bold.nii")
                if os.path.exists(c) and os.path.exists(r):
                    found[run_n] = (c, r)

            if found:
                for run_n in sorted(found):
                    c, r = found[run_n]
                    cartoon_files.append(c)
                    rest_files.append(r)
                    run_labels.append((f"{ses_tag}c0{run_n}", f"{ses_tag}r0{run_n}"))
                break

    return cartoon_files, rest_files, run_labels


def _smooth(sig, w):
    if not w or w <= 1 or len(sig) < w:
        return sig.copy()
    return uniform_filter1d(sig.astype(float), size=w, mode='nearest')


# ── concatenated analysis helpers (identical logic to adult script) ────────────

def _build_records(dysco_folder, run_labels, pid):
    all_npy = glob.glob(os.path.join(dysco_folder, "*_dysco.npy"))

    def _stem_to_key(fname):
        m = re.search(r"(ses-\d+).*task-(cartoon|rest).*run-(\d+)", fname)
        if m:
            return m.group(1), m.group(2), m.group(3)
        return None

    npy_index = {}
    for fpath in all_npy:
        key = _stem_to_key(os.path.basename(fpath))
        if key:
            npy_index[key] = fpath

    for fpath in all_npy:
        fname = os.path.basename(fpath)
        m = re.search(r"task-(cartoon|rest).*run-(\d+)", fname)
        if m and not re.search(r"ses-\d+", fname):
            npy_index[("ses-01", m.group(1), m.group(2))] = fpath

    records = []
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

    session_boundaries = set()
    prev_ses, bpos = None, 0
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
        ax.axvline(s, color="#aaaaaa", linestyle="--", linewidth=0.8, alpha=0.7, zorder=1)
        ax.text((s + e) / 2, 0.985, lbl,
                transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, color="#333333")

    for xpos in session_boundaries:
        ax.axvline(xpos, color="#333333", linestyle="-", linewidth=1.5, alpha=0.6, zorder=2)

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
                 mpatches.Patch(facecolor=C_REST,    label="Rest")],
        loc="upper right", fontsize=9, framealpha=0.9, edgecolor="#cccccc"
    )
    plt.tight_layout()
    path = os.path.join(out_dir, f"{pshort}_{measure}_concatenated{tag}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {os.path.basename(path)}")


def _condition_stats(records, pid, pshort, out_dir):
    rows = []
    for measure in ["entropy", "speed", "norm2"]:
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


def _run_concatenated_analysis(pid, pshort, run_labels, out_root, npy_folder=None):
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


def _get_ses1_signal(records, measure):
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
    os.makedirs(out_dir, exist_ok=True)
    pids = list(all_records_by_pid.keys())
    n    = len(pids)

    first_ses1 = [r for r in next(iter(all_records_by_pid.values()))
                  if r["session"] == "ses-01"]

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

        fig, axes = plt.subplots(n_panels, 1, figsize=(18, 4 * n_panels), sharex=False)
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
                         mpatches.Patch(facecolor=C_REST,    label="Rest")],
                loc="upper right", fontsize=8.5, framealpha=0.9, edgecolor="#cccccc"
            )

        axes[-1].set_xlabel("Time windows across consecutive runs", fontsize=11)
        fig.suptitle(
            f"Group Average DySCo Metrics — Healthy Controls  |  "
            f"Session 1  |  n = {n}  |  mean ± SD  ({suffix})",
            fontsize=13, fontweight="bold", y=1.01
        )
        plt.tight_layout()
        path = os.path.join(out_dir, f"hc_group_all_metrics_concatenated{tag}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {os.path.basename(path)}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    all_pids = sorted([
        d for d in os.listdir(BASE_DIR)
        if d.startswith("sub-hc") and "dysco" not in d
        and os.path.isdir(os.path.join(BASE_DIR, d))
    ])

    print(f"\n{'='*62}")
    print(f"  HC DySCo Pipeline — {len(all_pids)} participants found")
    print(f"  Output root: {OUTPUT_ROOT}")
    print(f"{'='*62}\n")

    cohort_config  = {}
    run_labels_map = {}
    skipped        = []

    for pid in all_pids:
        cartoon_files, rest_files, run_labels = _find_participant_runs(BASE_DIR, pid)
        run_labels_map[pid] = run_labels

        if pid in ALREADY_PROCESSED:
            print(f"  {pid}: already processed — will load existing .npy outputs.")
            continue

        if not cartoon_files:
            print(f"  WARNING: no NIfTI files found for {pid} — skipping.")
            skipped.append(pid)
            continue

        cohort_config[pid] = {
            "cartoon": cartoon_files,
            "rest": rest_files
        }
        print(f"  {pid}: queued for processing ({len(cartoon_files)} cartoon + {len(rest_files)} rest runs)")

    if skipped:
        print(f"\n  Skipped (missing data): {skipped}")

    if not cohort_config and not ALREADY_PROCESSED:
        print("ERROR: No participants found. Check BASE_DIR.")
        sys.exit(1)

    # ── Step 1: DySCo pipeline ────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  STEP 1/4 — DySCo pipeline ({len(cohort_config)} new participants)")
    print(f"{'='*62}\n")

    if cohort_config:
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
    print("  STEP 2/4 — Per-participant concatenated timecourses")
    print(f"{'='*62}\n")

    all_records_by_pid = {}

    # First load the already processed participants
    for pid, npy_folder in ALREADY_PROCESSED.items():
        pshort  = pid.replace("sub-", "")
        run_lbl = run_labels_map.get(pid, [])
        records = _run_concatenated_analysis(pid, pshort, run_lbl,
                                             OUTPUT_ROOT, npy_folder=npy_folder)
        if records:
            all_records_by_pid[pid] = records

    # Then load newly processed participants
    for pid in cohort_config:
        pshort = pid.replace("sub-", "")
        records = _run_concatenated_analysis(pid, pshort, run_labels_map[pid],
                                             OUTPUT_ROOT)
        if records:
            all_records_by_pid[pid] = records

    # ── Step 3: group average ─────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  STEP 3/4 — Concatenated group average (n={len(all_records_by_pid)}, ses-01)")
    print(f"{'='*62}\n")

    group_dir = os.path.join(OUTPUT_ROOT, "group_average")
    if all_records_by_pid:
        _plot_group_average(all_records_by_pid, group_dir)

    # ── Step 4: pipeline-style group figures ─────────────────────────────────
    print(f"\n{'='*62}")
    print("  STEP 4/4 — Pipeline-style group figures (cartoon / rest / wait)")
    print(f"{'='*62}\n")

    pipeline_data = {}
    for pid, records in all_records_by_pid.items():
        cartoon = [(r["label"], r["data"]) for r in records
                   if r["run_type"] == "c" and r["session"] == "ses-01"]
        rest    = [(r["label"], r["data"]) for r in records
                   if r["run_type"] == "r" and r["session"] == "ses-01"]
        if cartoon or rest:
            pipeline_data[pid] = {"cartoon": cartoon, "rest": rest}

    if pipeline_data:
        print(f"  Generating pipeline figures for {len(pipeline_data)} participants...")
        for fn, label in [
            (plot_group_cartoon,     "cartoon"),
            (plot_group_rest,        "rest"),
            (plot_group_wait_aligned,"wait-aligned")
        ]:
            try:
                fn(pipeline_data, group_dir, TR=TR)
            except Exception as e:
                print(f"  WARNING: {label} figure failed — {e}")

    print(f"\n{'='*62}")
    print("  All done.")
    print(f"  Individual outputs : {OUTPUT_ROOT}/<pid>_dysco_output/")
    print(f"  Group average      : {group_dir}")
    print(f"  Included in group  : {len(all_records_by_pid)} participants")
    print(f"{'='*62}\n")