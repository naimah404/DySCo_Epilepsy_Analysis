import os
import re
import numpy as np
import matplotlib.pyplot as plt

plt.ioff()


def extract_run_info(filename):
    match = re.search(r'_([cr])(\d+)_', filename.lower())
    if match:
        return match.group(1), int(match.group(2))
    return None, None


def load_sorted_records(folder):
    files = [f for f in os.listdir(folder) if f.endswith("_dysco.npy")]

    records = []
    for f in files:
        run_type, run_number = extract_run_info(f)
        if run_type is None:
            continue

        data = np.load(os.path.join(folder, f), allow_pickle=True).item()

        records.append({
            "filename": f,
            "run_type": run_type,
            "run_number": run_number,
            "data": data
        })

    # sort by run number
    records = sorted(records, key=lambda x: x["run_number"])
    return records


def concatenate_measure(records, measure):
    concatenated = []
    boundaries = []
    labels = []

    current_idx = 0

    for rec in records:
        data = rec["data"]

        if measure not in data:
            continue

        arr = np.asarray(data[measure], dtype=float)

        if arr.ndim != 1:
            continue

        concatenated.append(arr)

        start = current_idx
        end = current_idx + len(arr)

        boundaries.append((start, end, rec["run_type"]))
        labels.append(f"{rec['run_type']}{rec['run_number']:03d}")

        current_idx = end

    if not concatenated:
        return None, None, None

    return np.concatenate(concatenated), boundaries, labels


def plot_concatenated_measure(records, measure, output_folder, normalize=False):
    signal, boundaries, labels = concatenate_measure(records, measure)

    if signal is None:
        print(f"No valid {measure} found.")
        return

    if normalize:
        mn, mx = np.min(signal), np.max(signal)
        if mx > mn:
            signal = (signal - mn) / (mx - mn)

    fig, ax = plt.subplots(figsize=(14, 6))

    # plot continuous line
    ax.plot(signal, color="black", linewidth=1.5)

    # mark boundaries + shading
    for (start, end, run_type) in boundaries:
        color = "blue" if run_type == "c" else "red"

        ax.axvspan(start, end, color=color, alpha=0.1)

        # vertical boundary line
        ax.axvline(start, color="gray", linestyle="--", alpha=0.5)

    ax.set_title(f"P001 concatenated {measure}")
    ax.set_xlabel("Time Windows (concatenated runs)")
    ax.set_ylabel(f"{'Normalized ' if normalize else ''}{measure}")

    plt.tight_layout()

    norm_tag = "_normalized" if normalize else ""
    save_path = os.path.join(output_folder, f"p001_{measure}_concatenated{norm_tag}.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {save_path}")


def process_p001_concatenated():
    input_folder = r"C:/Users/naima/DySCo-main/DySCo-main/DELFT_NEW/P001_dysco_output"
    output_folder = r"C:/Users/naima/DySCo-main/DySCo-main/DELFT_NEW/P001/concatenated_runs"
    os.makedirs(output_folder, exist_ok=True)

    records = load_sorted_records(input_folder)

    print("Run order:")
    for r in records:
        print(f"{r['run_type']}{r['run_number']:03d}")

    for measure in ["entropy", "speed", "norm1", "norm2"]:
        plot_concatenated_measure(records, measure, output_folder, normalize=False)
        plot_concatenated_measure(records, measure, output_folder, normalize=True)

    print("\nDone.")


if __name__ == "__main__":
    process_p001_concatenated()