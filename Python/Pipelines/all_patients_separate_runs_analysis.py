import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.ioff()

try:
    from scipy.stats import ttest_ind
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


def extract_run_info(filename):
    """
    Extract run type and run number from names like:
    p001_c003_merged_dysco.npy
    p001_r004_merged_dysco.npy
    """
    match = re.search(r'_([cr])(\d+)_', filename.lower())
    if match:
        return match.group(1), int(match.group(2))
    return None, None


def smooth_signal(signal, window=5):
    """
    Simple moving average smoothing.
    """
    signal = np.asarray(signal, dtype=float)
    if window is None or window <= 1:
        return signal.copy()
    if len(signal) < window:
        return signal.copy()
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode="same")


def load_sorted_records(folder):
    """
    Load all *_dysco.npy files from a folder and sort by run number.
    """
    if not os.path.exists(folder):
        print(f"Folder not found: {folder}")
        return []

    files = [f for f in os.listdir(folder) if f.endswith("_dysco.npy")]

    records = []
    for f in files:
        run_type, run_number = extract_run_info(f)
        if run_type is None:
            print(f"Skipping unrecognized filename: {f}")
            continue

        full_path = os.path.join(folder, f)

        try:
            data = np.load(full_path, allow_pickle=True).item()
        except Exception as e:
            print(f"Could not load {full_path}: {e}")
            continue

        records.append({
            "filename": f,
            "run_type": run_type,
            "run_number": run_number,
            "condition": "cartoon" if run_type == "c" else "rest",
            "data": data
        })

    records = sorted(records, key=lambda x: x["run_number"])
    return records


def concatenate_measure(records, measure):
    """
    Concatenate a 1D measure across runs in sorted order.
    Returns:
    - concatenated signal
    - boundaries: list of (start, end, run_type, run_number)
    """
    concatenated = []
    boundaries = []
    current_idx = 0

    for rec in records:
        data = rec["data"]

        if measure not in data:
            print(f"{measure} not found in {rec['filename']}, skipping")
            continue

        arr = np.asarray(data[measure], dtype=float)

        if arr.ndim != 1 or len(arr) == 0:
            print(f"{measure} in {rec['filename']} is not valid 1D data, skipping")
            continue

        start = current_idx
        end = current_idx + len(arr)

        concatenated.append(arr)
        boundaries.append((start, end, rec["run_type"], rec["run_number"]))
        current_idx = end

    if not concatenated:
        return None, None

    return np.concatenate(concatenated), boundaries


def plot_concatenated_measure(records, measure, output_folder, patient_id, smooth_window=None):
    """
    Plot one continuous concatenated measure for a patient.
    Uses raw values, optionally smoothed.
    """
    signal, boundaries = concatenate_measure(records, measure)

    if signal is None:
        print(f"No valid {measure} found for {patient_id}.")
        return

    if smooth_window is not None and smooth_window > 1:
        signal_to_plot = smooth_signal(signal, window=smooth_window)
        smooth_tag = f"_smoothed_w{smooth_window}"
        title_suffix = f" (smoothed, window={smooth_window})"
    else:
        signal_to_plot = signal
        smooth_tag = "_raw"
        title_suffix = " (raw)"

    fig, ax = plt.subplots(figsize=(15, 6))

    for (start, end, run_type, run_number) in boundaries:
        color = "blue" if run_type == "c" else "red"
        label = f"{run_type}{run_number:03d}"

        ax.axvspan(start, end, color=color, alpha=0.10)
        ax.axvline(start, color="gray", linestyle="--", alpha=0.5, linewidth=1)

        midpoint = (start + end) / 2
        ax.text(
            midpoint,
            0.98,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9
        )

    if boundaries:
        ax.axvline(boundaries[-1][1], color="gray", linestyle="--", alpha=0.5, linewidth=1)

    ax.plot(signal_to_plot, color="black", linewidth=1.5)

    ax.set_title(f"{patient_id} concatenated {measure}{title_suffix}", fontsize=14)
    ax.set_xlabel("Time windows across consecutive runs", fontsize=12)
    ax.set_ylabel(measure, fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(
        output_folder,
        f"{patient_id.lower()}_{measure}_concatenated{smooth_tag}.png"
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def compute_condition_arrays(records, measure):
    """
    Collect raw 1D values by condition across runs for one patient.
    """
    cartoon_vals = []
    rest_vals = []

    for rec in records:
        data = rec["data"]
        if measure not in data:
            continue

        arr = np.asarray(data[measure], dtype=float)
        if arr.ndim != 1 or len(arr) == 0:
            continue

        if rec["run_type"] == "c":
            cartoon_vals.append(arr)
        elif rec["run_type"] == "r":
            rest_vals.append(arr)

    cartoon_vals = np.concatenate(cartoon_vals) if cartoon_vals else np.array([])
    rest_vals = np.concatenate(rest_vals) if rest_vals else np.array([])

    return cartoon_vals, rest_vals


def compute_condition_stats(records, patient_id):
    """
    Compute raw condition stats for entropy, speed, norm1, norm2.
    """
    rows = []

    for measure in ["entropy", "speed", "norm1", "norm2"]:
        cartoon_vals, rest_vals = compute_condition_arrays(records, measure)

        row = {
            "patient_id": patient_id,
            "measure": measure,
            "cartoon_n": len(cartoon_vals),
            "rest_n": len(rest_vals),
            "cartoon_mean": np.mean(cartoon_vals) if len(cartoon_vals) else np.nan,
            "cartoon_std": np.std(cartoon_vals) if len(cartoon_vals) else np.nan,
            "rest_mean": np.mean(rest_vals) if len(rest_vals) else np.nan,
            "rest_std": np.std(rest_vals) if len(rest_vals) else np.nan,
            "mean_difference_cartoon_minus_rest": (
                np.mean(cartoon_vals) - np.mean(rest_vals)
                if len(cartoon_vals) and len(rest_vals) else np.nan
            ),
            "t_statistic": np.nan,
            "p_value": np.nan
        }

        if SCIPY_AVAILABLE and len(cartoon_vals) > 1 and len(rest_vals) > 1:
            t_stat, p_val = ttest_ind(cartoon_vals, rest_vals, equal_var=False)
            row["t_statistic"] = t_stat
            row["p_value"] = p_val

        rows.append(row)

    return pd.DataFrame(rows)


def plot_condition_boxplot(records, measure, output_folder, patient_id):
    """
    Boxplot comparing cartoon vs rest for one measure, one patient.
    """
    cartoon_vals, rest_vals = compute_condition_arrays(records, measure)

    if len(cartoon_vals) == 0 or len(rest_vals) == 0:
        print(f"Not enough data for boxplot: {patient_id} {measure}")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.boxplot([cartoon_vals, rest_vals], tick_labels=["Cartoon", "Rest"])
    ax.set_title(f"{patient_id} {measure}: cartoon vs rest", fontsize=14)
    ax.set_ylabel(measure, fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(
        output_folder,
        f"{patient_id.lower()}_{measure}_cartoon_vs_rest_boxplot.png"
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def save_run_order(records, output_folder, patient_id):
    save_path = os.path.join(output_folder, f"{patient_id.lower()}_run_order.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"Run order for {patient_id}\n")
        f.write("=" * 40 + "\n")
        for rec in records:
            f.write(
                f"{rec['run_type']}{rec['run_number']:03d} -> {rec['filename']} "
                f"({rec['condition']})\n"
            )
    print(f"Saved: {save_path}")


def save_run_level_summary(records, output_folder, patient_id):
    rows = []
    for rec in records:
        row = {
            "patient_id": patient_id,
            "filename": rec["filename"],
            "run_type": rec["run_type"],
            "run_number": rec["run_number"],
            "condition": rec["condition"],
            "metastability": rec["data"].get("metastability", np.nan)
        }

        for measure in ["entropy", "speed", "norm1", "norm2"]:
            if measure in rec["data"]:
                arr = np.asarray(rec["data"][measure], dtype=float)
                if arr.ndim == 1 and len(arr) > 0:
                    row[f"{measure}_mean"] = np.mean(arr)
                    row[f"{measure}_std"] = np.std(arr)
                    row[f"{measure}_min"] = np.min(arr)
                    row[f"{measure}_max"] = np.max(arr)
                    row[f"{measure}_len"] = len(arr)
                else:
                    row[f"{measure}_mean"] = np.nan
                    row[f"{measure}_std"] = np.nan
                    row[f"{measure}_min"] = np.nan
                    row[f"{measure}_max"] = np.nan
                    row[f"{measure}_len"] = np.nan
            else:
                row[f"{measure}_mean"] = np.nan
                row[f"{measure}_std"] = np.nan
                row[f"{measure}_min"] = np.nan
                row[f"{measure}_max"] = np.nan
                row[f"{measure}_len"] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("run_number")
    save_path = os.path.join(output_folder, f"{patient_id.lower()}_run_level_summary.csv")
    df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")
    return df


def process_patient(patient_id, base_folder):
    input_folder = os.path.join(base_folder, f"{patient_id}_dysco_output")
    output_folder = os.path.join(base_folder, patient_id, "concatenated_runs_raw_with_stats")
    os.makedirs(output_folder, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"Processing {patient_id}")
    print("=" * 70)

    records = load_sorted_records(input_folder)

    if not records:
        print(f"No valid records found for {patient_id}")
        return None, None

    print("Run order:")
    for r in records:
        print(f"  {r['run_type']}{r['run_number']:03d}")

    save_run_order(records, output_folder, patient_id)
    save_run_level_summary(records, output_folder, patient_id)

    for measure in ["entropy", "speed", "norm1", "norm2"]:
        plot_concatenated_measure(records, measure, output_folder, patient_id, smooth_window=None)
        plot_concatenated_measure(records, measure, output_folder, patient_id, smooth_window=5)
        plot_condition_boxplot(records, measure, output_folder, patient_id)

    stats_df = compute_condition_stats(records, patient_id)
    stats_path = os.path.join(output_folder, f"{patient_id.lower()}_condition_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved: {stats_path}")

    return records, stats_df


def main():
    base_folder = r"C:/Users/naima/DySCo-main/DySCo-main/DELFT_NEW"
    patients = ["P001", "P002", "P003", "P004", "P005"]

    all_stats = []

    for patient_id in patients:
        _, stats_df = process_patient(patient_id, base_folder)
        if stats_df is not None:
            all_stats.append(stats_df)

    if all_stats:
        combined_stats = pd.concat(all_stats, ignore_index=True)
        combined_path = os.path.join(base_folder, "all_patients_condition_stats_raw.csv")
        combined_stats.to_csv(combined_path, index=False)
        print(f"\nSaved combined stats: {combined_path}")

    print("\nAll patients complete.")


if __name__ == "__main__":
    main()