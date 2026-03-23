import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================

data_folder = r"C:/Users/naima/DySCo-main/DySCo-main/DELFT_NEW/P001_dysco_output"

# metrics to plot
metrics = ["entropy", "speed", "norm2"]

# ============================================================
# GET ALL DYSC0 FILES
# ============================================================

files = sorted(glob.glob(os.path.join(data_folder, "*_dysco.npy")))

print("Found files:", len(files))

runs = []

# ============================================================
# LOAD FILES
# ============================================================

for f in files:

    name = os.path.basename(f)

    # detect condition from filename
    if "_c" in name:
        condition = "cartoon"
    elif "_r" in name:
        condition = "rest"
    else:
        print("Skipping:", name)
        continue

    # load dictionary
    data = np.load(f, allow_pickle=True).item()

    runs.append({
        "name": name,
        "condition": condition,
        "entropy": np.array(data["entropy"]).squeeze(),
        "speed": np.array(data["speed"]).squeeze(),
        "norm2": np.array(data["norm2"]).squeeze()
    })

print("Loaded runs:", len(runs))

# ============================================================
# PLOT ALL RUNS
# ============================================================

for metric in metrics:

    plt.figure(figsize=(10,5))

    for run in runs:

        y = run[metric]
        x = np.arange(len(y))

        if run["condition"] == "cartoon":
            plt.plot(x, y, label=run["name"], linestyle="-")
        else:
            plt.plot(x, y, label=run["name"], linestyle="--")

    plt.title(f"P001 {metric} across runs")
    plt.xlabel("Time / Window")
    plt.ylabel(metric)
    plt.grid(True)
    plt.legend(fontsize=8)

    plt.show()