import os
import numpy as np
import matplotlib.pyplot as plt

plt.ioff()


def plot_epilepsy_group_entropy(
    participant_folders,
    normalize=True,
    save_plot=True,
    save_path=None
):
    """
    Plot one epilepsy group-average entropy curve with std shading.

    Workflow:
    1. Within each patient folder, load all *_dysco.npy files
    2. Extract entropy from each file
    3. Average across files within that patient
    4. Average across patients
    5. Plot mean ± std

    Parameters
    ----------
    participant_folders : list of str
        List of epilepsy patient output folders, e.g. P002_dysco_output, ...
    normalize : bool
        If True, min-max normalize each entropy curve before averaging
    save_plot : bool
        If True, save the plot
    save_path : str or None
        Full output path for the figure, or directory if preferred
    """

    patient_curves = []

    for folder_path in participant_folders:
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            continue

        all_files = sorted(
            f for f in os.listdir(folder_path)
            if f.endswith("_dysco.npy")
        )

        if not all_files:
            print(f"No *_dysco.npy files found in {folder_path}")
            continue

        file_entropies = []

        for filename in all_files:
            full_path = os.path.join(folder_path, filename)

            try:
                data = np.load(full_path, allow_pickle=True).item()
            except Exception as e:
                print(f"Could not load {full_path}: {e}")
                continue

            if "entropy" in data:
                entropy = np.asarray(data["entropy"], dtype=float)
            elif "von_neumann_entropy" in data:
                entropy = np.asarray(data["von_neumann_entropy"], dtype=float)
            else:
                print(f"No entropy key in {filename}, skipping")
                continue

            if entropy.ndim != 1 or len(entropy) == 0:
                print(f"Invalid entropy shape in {filename}, skipping")
                continue

            if normalize:
                mn = np.min(entropy)
                mx = np.max(entropy)
                if mx > mn:
                    entropy = (entropy - mn) / (mx - mn)
                else:
                    entropy = np.zeros_like(entropy)

            file_entropies.append(entropy)

        if not file_entropies:
            print(f"No valid entropy data found for {folder_path}")
            continue

        # Match lengths within this patient before averaging files
        min_len_patient = min(len(arr) for arr in file_entropies)
        patient_matrix = np.vstack([arr[:min_len_patient] for arr in file_entropies])

        # One entropy curve per patient
        patient_mean_curve = np.mean(patient_matrix, axis=0)
        patient_curves.append(patient_mean_curve)

        print(f"Added patient from {folder_path} with {len(file_entropies)} files")

    if not patient_curves:
        print("No patient entropy curves available.")
        return

    # Match lengths across patients before group averaging
    min_len_group = min(len(arr) for arr in patient_curves)
    group_matrix = np.vstack([arr[:min_len_group] for arr in patient_curves])

    group_mean = np.mean(group_matrix, axis=0)
    group_std = np.std(group_matrix, axis=0)

    x = np.arange(min_len_group)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        x,
        group_mean,
        color="red",
        linewidth=2.5,
        label=f"Epilepsy (n={len(patient_curves)})"
    )

    ax.fill_between(
        x,
        group_mean - group_std,
        group_mean + group_std,
        color="red",
        alpha=0.2
    )

    ax.set_xlabel("Time Windows", fontsize=12)
    ax.set_ylabel("Normalized Entropy" if normalize else "Entropy", fontsize=12)
    ax.set_title("Epilepsy Group Average Entropy", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plot:
        if save_path is None:
            save_path = os.path.join(os.getcwd(), "epilepsy_group_entropy.png")
        elif os.path.isdir(save_path):
            save_path = os.path.join(save_path, "epilepsy_group_entropy.png")

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")

    plt.close(fig)
    return fig, ax

if __name__ == "__main__":
    participant_folders = [
        "C:/Users/naima/DySCo-main/DySCo-main/DELFT_NEW/P001_dysco_output/",
        "C:/Users/naima/DySCo-main/DySCo-main/DELFT_NEW/P002_dysco_output/",
        "C:/Users/naima/DySCo-main/DySCo-main/DELFT_NEW/P003_dysco_output/",
        "C:/Users/naima/DySCo-main/DySCo-main/DELFT_NEW/P004_dysco_output/",
        "C:/Users/naima/DySCo-main/DySCo-main/DELFT_NEW/P005_dysco_output/"
    ]

    plot_epilepsy_group_entropy(
        participant_folders=participant_folders,
        normalize=True,
        save_plot=True,
        save_path="C:/Users/naima/DySCo-main/DySCo-main/DELFT_NEW/"
    )