import os
import numpy as np
import nibabel as nb
# Redownload nibabel because it wont work ??
import sys
from pathlib import Path
import glob

# Add core functions to path
# Change path when needed 
core_functions_path = os.path.abspath('C:/Users/naima/DySCo-main/DySCo-main/Python/core_functions')
sys.path.append(core_functions_path)

# Import DySCo functions
from compute_eigenvectors_sliding_cov import compute_eigs_cov
from dysco_distance import dysco_distance
from dysco_norm import dysco_norm
from fMRI_Processing.surf_cifti_data import surf_data_from_cifti


def process_single_file(file_path, output_folder, half_window_size=10, n_eigen=10, lag=20):
    """
    Process a single fMRI file (.nii) and calculate all DySCo measures. 
    All DySCo measures are saved into a dictionary
    """
    print(f"Processing: {os.path.basename(file_path)}")
    
    # Step 1: Load the data
    if file_path.endswith('.nii'):
        nifti = nb.load(file_path)
        fmri_data = nifti.get_fdata()
        
        # Convert 4D to 2D if necessary
        if len(fmri_data.shape) == 4:
            brain_2d = fmri_data.reshape(-1, fmri_data.shape[-1]).T
        else:
            brain_2d = fmri_data
    elif file_path.endswith('.npy'):
        brain_2d = np.load(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # Preprocess
    brain_2d = np.nan_to_num(brain_2d)
    brain_2d = brain_2d.astype(np.float64)
    brain_2d = np.array(brain_2d)
    
    # Add tiny noise for regularization
    brain_2d_regularized = brain_2d + 1e-6 * np.random.randn(*brain_2d.shape)
    
    # Step 2: Compute eigenvectors and eigenvalues
    eigenvectors, eigenvalues = compute_eigs_cov(brain_2d_regularized, n_eigen, half_window_size)
    
    T = eigenvectors.shape[0]
    
    # Step 3: Calculate all DySCo measures
    
    # 3.1 Norms (all three types)
    norm1 = dysco_norm(eigenvalues, 1)
    norm2 = dysco_norm(eigenvalues, 2)
    
    
    # 3.2 Metastability (from norm)
    metastability = np.std(norm2)
    
    # 3.3 FCD Matrix (distance matrix)
    fcd = np.zeros((T, T))
    for i in range(T):
        for j in range(i + 1, T):
            fcd[i, j] = dysco_distance(eigenvectors[i, :, :], eigenvectors[j, :, :], 2)
            fcd[j, i] = fcd[i, j]
    
    # 3.4 Reconfiguration speed
    speed = np.zeros(T - lag)
    for i in range(T - lag):
        speed[i] = fcd[i, i + lag]
    
    # 3.5 Von Neumann Entropy
    eigenvalues_norm = eigenvalues / np.tile(np.sum(eigenvalues, axis=0), (n_eigen, 1))
    entropy = -np.sum(np.log(eigenvalues_norm + 1e-10) * eigenvalues_norm, axis=0)
    
    # Compile all measures into a dictionary
    dysco_measures = {
        'filename': os.path.basename(file_path),
        'norm1': norm1,
        #we're only using norm2 so far 
        'norm2': norm2,
        
        'metastability': metastability,
        'speed': speed,
        'entropy': entropy,
        'fcd': fcd,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors
    }
    
    # Save individual file results
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = os.path.join(output_folder, f"{base_name}_dysco.npy")
    np.save(output_file, dysco_measures)
    print(f"  Saved to: {output_file}")
    
    return dysco_measures


def batch_process_folder(input_folder, output_folder, file_pattern='*.nii', 
                         half_window_size=10, n_eigen=10, lag=20):
    """
    Process all files in a folder and save individual dictionaries.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all files matching pattern
    file_list = glob.glob(os.path.join(input_folder, file_pattern))
    
    if not file_list:
        print(f"No files found matching pattern: {file_pattern} in {input_folder}")
        return
    
    print(f"Found {len(file_list)} files to process")
    print(f"Output folder: {output_folder}")
    print("-" * 50)
    
    # Process each file
    for file_path in file_list:
        try:
            process_single_file(
                file_path, 
                output_folder,
                half_window_size=half_window_size,
                n_eigen=n_eigen,
                lag=lag
            )
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    print("-" * 50)
    print(f"Processing complete! Files saved to: {output_folder}")


# Example usage - single if __name__ block with all participants
if __name__ == "__main__":
    
    # List of participants to process
    participants = ['P002', 'P003', 'P004', 'P005']
    
    # Parameters that were in the simple tutorial??
    half_window_size = 10  # Half window size (total window = 2*half_window_size + 1)
    n_eigen = 10           # Number of eigenvectors
    lag = 20               # Lag for reconfiguration speed
    
    # Process each participant
    for participant in participants:
        print(f"\n{'='*60}")
        print(f"Processing {participant}")
        print('='*60)
        
        input_folder = f"C:/Users/naima/DySCo-main/DySCo-main/DELFT_NEW/{participant}"
        output_folder = f"C:/Users/naima/DySCo-main/DySCo-main/DELFT_NEW/{participant}_dysco_output"
        
        # Run batch processing
        batch_process_folder(
            input_folder=input_folder,
            output_folder=output_folder,
            file_pattern='*.nii',  # or '*.npy' for numpy files
            half_window_size=half_window_size,
            n_eigen=n_eigen,
            lag=lag
        )