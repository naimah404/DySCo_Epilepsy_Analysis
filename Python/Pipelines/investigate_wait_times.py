import os
import glob
import json
import nibabel as nb
import numpy as np
import pandas as pd


def inspect_single_nifti(file_path, search_sidecars=True, preview_json=True):
    """
    Inspect one NIfTI file and return a dictionary of useful metadata.
    """
    result = {
        "file_path": file_path,
        "file_name": os.path.basename(file_path),
        "exists": os.path.exists(file_path),
        "shape": None,
        "n_volumes": None,
        "tr_seconds": None,
        "total_duration_seconds": None,
        "total_duration_minutes": None,
        "spatial_units": None,
        "time_units": None,
        "sidecar_files_found": [],
        "json_sidecar_found": None,
        "json_keys_preview": {},
        "status": "ok",
        "error": None,
    }

    print("\n" + "=" * 90)
    print(f"INSPECTING: {file_path}")
    print("=" * 90)

    if not os.path.exists(file_path):
        print("File not found.")
        result["status"] = "missing"
        result["error"] = "File not found"
        return result

    try:
        img = nb.load(file_path)
        data = img.get_fdata()
        hdr = img.header
    except Exception as e:
        print(f"Could not load file: {e}")
        result["status"] = "load_failed"
        result["error"] = str(e)
        return result

    # Basic info
    shape = data.shape
    result["shape"] = shape
    print("\n--- BASIC INFO ---")
    print(f"Shape: {shape}")
    print(f"Datatype: {data.dtype}")

    if data.ndim == 4:
        n_volumes = shape[3]
        result["n_volumes"] = int(n_volumes)
        print(f"Number of volumes: {n_volumes}")
    else:
        print("This file is not 4D.")
        n_volumes = None

    zooms = hdr.get_zooms()
    print(f"Header zooms: {zooms}")

    tr = None
    if len(zooms) >= 4:
        try:
            tr = float(zooms[3])
        except Exception:
            tr = None

    if tr is not None:
        result["tr_seconds"] = tr
        print(f"TR: {tr:.6f} seconds")
    else:
        print("TR not found in header.")

    if n_volumes is not None and tr is not None and tr > 0:
        total_duration_sec = n_volumes * tr
        result["total_duration_seconds"] = float(total_duration_sec)
        result["total_duration_minutes"] = float(total_duration_sec / 60.0)
        print(f"Total duration: {total_duration_sec:.2f} seconds")
        print(f"Total duration: {total_duration_sec / 60.0:.2f} minutes")
    else:
        print("Total duration could not be computed.")

    # Header info
    print("\n--- HEADER INFO ---")
    xyzt_units = hdr.get_xyzt_units()
    result["spatial_units"] = xyzt_units[0]
    result["time_units"] = xyzt_units[1]
    print(f"Spatial units: {xyzt_units[0]}")
    print(f"Time units: {xyzt_units[1]}")
    print(f"pixdim: {hdr['pixdim']}")
    print(f"slice_duration: {hdr['slice_duration']}")
    print(f"toffset: {hdr['toffset']}")

    # Sidecars
    if search_sidecars:
        print("\n--- POSSIBLE SIDECAR / TIMING FILES ---")
        folder = os.path.dirname(file_path)
        base = os.path.basename(file_path)

        if base.endswith(".nii.gz"):
            stem = base[:-7]
        else:
            stem = os.path.splitext(base)[0]

        patterns = [
            os.path.join(folder, f"{stem}.json"),
            os.path.join(folder, f"{stem}.tsv"),
            os.path.join(folder, f"{stem}.csv"),
            os.path.join(folder, f"{stem}.txt"),
            os.path.join(folder, "*events*"),
            os.path.join(folder, "*design*"),
            os.path.join(folder, "*stim*"),
            os.path.join(folder, "*task*"),
            os.path.join(folder, "*.json"),
            os.path.join(folder, "*.tsv"),
            os.path.join(folder, "*.csv"),
            os.path.join(folder, "*.txt"),
        ]

        found = []
        for pattern in patterns:
            found.extend(glob.glob(pattern))

        found = sorted(set(found))
        result["sidecar_files_found"] = found

        if found:
            for f in found:
                print(f)
        else:
            print("No obvious timing/design sidecar files found in the same folder.")

        json_sidecar = os.path.join(folder, f"{stem}.json")
        if os.path.exists(json_sidecar):
            result["json_sidecar_found"] = json_sidecar
            print("\n--- JSON SIDECAR FOUND ---")
            print(json_sidecar)

            if preview_json:
                try:
                    with open(json_sidecar, "r", encoding="utf-8") as f:
                        meta = json.load(f)

                    keys_of_interest = [
                        "RepetitionTime",
                        "TaskName",
                        "SliceTiming",
                        "AcquisitionDuration",
                        "Instructions",
                        "ProtocolName",
                        "SeriesDescription",
                    ]

                    preview = {}
                    for key in keys_of_interest:
                        if key in meta:
                            preview[key] = meta[key]
                            print(f"{key}: {meta[key]}")

                    result["json_keys_preview"] = preview

                except Exception as e:
                    print(f"Could not read JSON sidecar: {e}")

    print("\n--- INTERPRETATION ---")
    if data.ndim != 4:
        print("Not a 4D fMRI time-series file.")
    elif tr is None or tr == 0:
        print("4D file found, but TR is missing or zero.")
    else:
        print("4D file found with usable TR.")

    if result["sidecar_files_found"]:
        print("Possible timing/design-related files exist nearby.")
    else:
        print("No nearby timing/design files were detected.")

    return result


def inspect_all_niftis(root_folder, save_csv=True, save_txt=True):
    """
    Find and inspect all .nii and .nii.gz files under a root folder.
    """
    nii_files = glob.glob(os.path.join(root_folder, "**", "*.nii"), recursive=True)
    nii_gz_files = glob.glob(os.path.join(root_folder, "**", "*.nii.gz"), recursive=True)

    all_files = sorted(set(nii_files + nii_gz_files))

    if not all_files:
        print(f"No NIfTI files found under: {root_folder}")
        return None

    print(f"\nFound {len(all_files)} NIfTI files under:\n{root_folder}")

    results = []
    for i, file_path in enumerate(all_files, start=1):
        print(f"\n[{i}/{len(all_files)}]")
        result = inspect_single_nifti(file_path)
        results.append(result)

    # Build summary dataframe
    summary_rows = []
    for r in results:
        summary_rows.append({
            "file_name": r["file_name"],
            "file_path": r["file_path"],
            "shape": str(r["shape"]),
            "n_volumes": r["n_volumes"],
            "tr_seconds": r["tr_seconds"],
            "total_duration_seconds": r["total_duration_seconds"],
            "total_duration_minutes": r["total_duration_minutes"],
            "spatial_units": r["spatial_units"],
            "time_units": r["time_units"],
            "n_sidecar_files_found": len(r["sidecar_files_found"]),
            "json_sidecar_found": r["json_sidecar_found"],
            "status": r["status"],
            "error": r["error"],
        })

    df = pd.DataFrame(summary_rows)

    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    print(df.to_string(index=False))

    # Save outputs
    if save_csv:
        csv_path = os.path.join(root_folder, "nifti_inspection_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSaved CSV summary to: {csv_path}")

    if save_txt:
        txt_path = os.path.join(root_folder, "nifti_inspection_summary.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("NIFTI INSPECTION SUMMARY\n")
            f.write("=" * 90 + "\n\n")
            f.write(df.to_string(index=False))
            f.write("\n\nDETAILED SIDECAR FILES\n")
            f.write("=" * 90 + "\n\n")

            for r in results:
                f.write(f"FILE: {r['file_path']}\n")
                f.write(f"SIDECARS FOUND: {len(r['sidecar_files_found'])}\n")
                for s in r["sidecar_files_found"]:
                    f.write(f"  {s}\n")
                f.write("\n")

        print(f"Saved TXT summary to: {txt_path}")

    return df, results


if __name__ == "__main__":
    root_folder = r"C:/Users/naima/DySCo-main/DySCo-main/DELFT_NEW/"
    df, results = inspect_all_niftis(root_folder)