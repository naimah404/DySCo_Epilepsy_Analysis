import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Turn off interactive display
plt.ioff()

def plot_fcd_single(file_path, output_folder=None, cmap='viridis', 
                    title=None, figsize=(10, 8), dpi=150, show_colorbar=True):
    """
    Plot FCD matrix from a single DySCo file.
    """
    # Load data
    data = np.load(file_path, allow_pickle=True).item()
    
    # Get FCD matrix
    if "fcd" in data:
        fcd = data["fcd"]
    else:
        print(f"Error: No FCD matrix found in {file_path}")
        return
    
    # Get filename for title/label
    filename = os.path.basename(file_path).replace('_dysco.npy', '')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot FCD matrix
    im = ax.imshow(fcd, cmap=cmap, aspect='auto', origin='lower', interpolation='nearest')
    
    # Add colorbar
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Distance', fontsize=10)
    
    # Labels and title
    ax.set_xlabel('Time Windows', fontsize=12)
    ax.set_ylabel('Time Windows', fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'FCD Matrix - {filename}', fontsize=14)
    
    # Determine output path
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        save_path = os.path.join(output_folder, f"{filename}_fcd.png")
    else:
        save_path = os.path.join(os.path.dirname(file_path), f"{filename}_fcd.png")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved FCD plot to: {save_path}")
    
    return fig, ax


def plot_fcd_all_files(folder_path, output_folder=None, cmap='viridis', 
                       figsize=(10, 8), dpi=150, show_colorbar=True):
    """
    Plot FCD matrices for all *_dysco.npy files in a folder as separate plots.
    """
    # Find all files
    all_files = [f for f in os.listdir(folder_path) if f.endswith('_dysco.npy')]
    
    if not all_files:
        print(f"No *_dysco.npy files found in {folder_path}")
        return
    
    print(f"Found {len(all_files)} files to plot FCD matrices")
    print("-" * 50)
    
    # Set output folder
    if output_folder is None:
        output_folder = folder_path
    os.makedirs(output_folder, exist_ok=True)
    
    # Plot each file
    for filename in all_files:
        file_path = os.path.join(folder_path, filename)
        plot_fcd_single(
            file_path, 
            output_folder=output_folder,
            cmap=cmap, 
            figsize=figsize, 
            dpi=dpi,
            show_colorbar=show_colorbar
        )
    
    print("-" * 50)
    print(f"All FCD plots saved to: {output_folder}")


def plot_fcd_grid(folder_path, output_folder=None, cmap='viridis', 
                  cols=2, figsize=(15, 12), dpi=150):
    """
    Plot FCD matrices from all files in a grid layout (multiple plots in one figure).
    """
    # Find all files
    all_files = [f for f in os.listdir(folder_path) if f.endswith('_dysco.npy')]
    
    if not all_files:
        print(f"No *_dysco.npy files found in {folder_path}")
        return
    
    n_files = len(all_files)
    rows = (n_files + cols - 1) // cols
    
    print(f"Creating FCD grid for {folder_path} with {rows} rows and {cols} columns")
    
    # Create figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if n_files > 1 else [axes]
    
    # Plot each file
    for i, filename in enumerate(all_files):
        file_path = os.path.join(folder_path, filename)
        data = np.load(file_path, allow_pickle=True).item()
        
        if "fcd" not in data:
            print(f"Warning: No FCD in {filename}, skipping")
            continue
        
        fcd = data["fcd"]
        label = filename.replace('_dysco.npy', '')
        
        # Plot in subplot
        im = axes[i].imshow(fcd, cmap=cmap, aspect='auto', origin='lower')
        axes[i].set_title(label, fontsize=10)
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel('Time')
    
    # Hide empty subplots
    for i in range(n_files, len(axes)):
        axes[i].set_visible(False)
    
    # Add colorbar
    plt.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label='Distance')
    
    # Set main title
    folder_name = os.path.basename(os.path.normpath(folder_path))
    plt.suptitle(f'FCD Matrices - {folder_name}', fontsize=16, y=1.02)
    
    plt.tight_layout()
    
    # Save
    if output_folder is None:
        output_folder = folder_path
    os.makedirs(output_folder, exist_ok=True)
    
    save_path = os.path.join(output_folder, f"fcd_grid_{folder_name}.png")
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved FCD grid to: {save_path}")
    
    return fig, axes


def plot_fcd_for_all_participants(base_folder="C:/Users/naima/DySCo-main/DySCo-main/DELFT_NEW/",
                                  participants=None, plot_type='both'):
    """
    Plot FCD matrices for all participants (P001 to P005).
    
    Parameters:
    -----------
    base_folder : str
        Base folder containing participant subfolders
    participants : list, optional
        List of participant folders (e.g., ['P001', 'P002', ...])
        If None, will try P001-P005
    plot_type : str
        'individual' - plot each file separately
        'grid' - plot grid for each participant
        'both' - do both
    """
    if participants is None:
        participants = [f"P00{i}" for i in range(1, 6)]  # P001, P002, P003, P004, P005
    
    print("\n" + "="*70)
    print("PLOTTING FCD MATRICES FOR ALL PARTICIPANTS")
    print("="*70)
    
    for participant in participants:
        folder_path = os.path.join(base_folder, f"{participant}_dysco_output")
        
        if not os.path.exists(folder_path):
            print(f"\nFolder not found: {folder_path}")
            continue
        
        print(f"\n{'='*50}")
        print(f"Processing {participant}")
        print('='*50)
        
        if plot_type in ['individual', 'both']:
            plot_fcd_all_files(
                folder_path=folder_path,
                output_folder=None,  # Save in same folder
                cmap='viridis',
                figsize=(10, 8),
                dpi=150
            )
        
        if plot_type in ['grid', 'both']:
            plot_fcd_grid(
                folder_path=folder_path,
                output_folder=None,
                cmap='viridis',
                cols=2,
                figsize=(15, 12),
                dpi=150
            )
    
    print("\n" + "="*70)
    print("FCD PLOTTING COMPLETE FOR ALL PARTICIPANTS!")
    print("="*70)


def create_comparison_across_participants(base_folder="C:/Users/naima/DySCo-main/DySCo-main/DELFT_NEW/",
                                         participants=None, output_file="fcd_all_participants_comparison.png"):
    """
    Create a single figure comparing FCD matrices from all participants.
    """
    if participants is None:
        participants = [f"P00{i}" for i in range(1, 6)]  # P001, P002, P003, P004, P005
    
    # Collect valid folders
    valid_folders = []
    valid_labels = []
    
    for participant in participants:
        folder_path = os.path.join(base_folder, f"{participant}_dysco_output")
        if os.path.exists(folder_path):
            # Get first file in folder
            files = [f for f in os.listdir(folder_path) if f.endswith('_dysco.npy')]
            if files:
                valid_folders.append(folder_path)
                valid_labels.append(participant)
    
    if not valid_folders:
        print("No valid folders found")
        return
    
    n_folders = len(valid_folders)
    cols = min(3, n_folders)
    rows = (n_folders + cols - 1) // cols
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    axes = axes.flatten() if n_folders > 1 else [axes]
    
    # Plot first file from each participant
    for i, (folder_path, label) in enumerate(zip(valid_folders, valid_labels)):
        # Get first file
        files = [f for f in os.listdir(folder_path) if f.endswith('_dysco.npy')]
        file_path = os.path.join(folder_path, files[0])
        
        data = np.load(file_path, allow_pickle=True).item()
        fcd = data["fcd"]
        
        # Plot
        im = axes[i].imshow(fcd, cmap='viridis', aspect='auto', origin='lower')
        axes[i].set_title(label, fontsize=12)
        axes[i].set_xlabel('Time Windows')
        axes[i].set_ylabel('Time Windows')
    
    # Hide empty subplots
    for i in range(n_folders, len(axes)):
        axes[i].set_visible(False)
    
    # Add colorbar
    plt.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label='Distance')
    
    plt.suptitle('FCD Matrix Comparison Across Participants', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(base_folder, output_file)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nSaved comparison plot to: {save_path}")
    
    return fig, axes


# Run plotting when script is executed directly
if __name__ == "__main__":

    # Base directory containing participant folders
    base_folder = "C:/Users/naima/DySCo-main/DySCo-main/DELFT_NEW/"

    # Plot FCD matrices for all participants (default: P001–P005)
    plot_fcd_for_all_participants(
        base_folder=base_folder,
        participants=None,      # If None, the function will use P001–P005
        plot_type='both'        # Options: 'individual', 'grid', or 'both'
    )

    # Create a combined comparison figure across participants
    create_comparison_across_participants(
        base_folder=base_folder,
        participants=None,      # Again defaults to P001–P005
        output_file="fcd_all_participants_comparison.png"
    )

    # If you only want to run specific participants, use something like this:
    """
    selected_participants = ['P002', 'P004']

    plot_fcd_for_all_participants(
        base_folder=base_folder,
        participants=selected_participants,
        plot_type='both'
    )
    """

    print("\nFCD plotting finished.")