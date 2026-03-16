import os
import numpy as np
import matplotlib.pyplot as plt

# Set matplotlib to not display plots interactively
plt.ioff()

def plot_all_entropy(folder_path, normalize=True, save_plot=True):
    """
    Load all *_dysco.npy files from a folder and plot their entropy on one plot.
    
    Parameters:
    -----------
    folder_path : str
        Path to the folder containing the *_dysco.npy files
    normalize : bool
        Whether to min-max normalize the data before plotting (default: True)
    save_plot : bool
        Whether to save the plot to the folder (default: True)
    """
    # Find all files ending with _dysco.npy
    all_files = [f for f in os.listdir(folder_path) if f.endswith('_dysco.npy')]
    
    if not all_files:
        print(f"No *_dysco.npy files found in {folder_path}")
        return
    
    print(f"Found {len(all_files)} files to plot for entropy")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Use a colormap to generate different colors for each file
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_files)))
    
    # Plot each file
    for i, filename in enumerate(all_files):
        full_path = os.path.join(folder_path, filename)
        
        # Load data
        data = np.load(full_path, allow_pickle=True).item()
        
        # Get entropy (handle different possible key names)
        if "entropy" in data:
            entropy = data["entropy"]
        elif "von_neumann_entropy" in data:
            entropy = data["von_neumann_entropy"]
        else:
            print(f"Warning: No entropy found in {filename}, skipping")
            continue
        
        # Normalize if requested
        if normalize:
            entropy = (entropy - np.min(entropy)) / (np.max(entropy) - np.min(entropy))
            ylabel = 'Normalized Entropy'
        else:
            ylabel = 'Von Neumann Entropy'
        
        # Use filename without '_dysco' as label
        label = filename.replace('_dysco.npy', '')
        
        # Plot
        ax.plot(entropy, color=colors[i], label=label, linewidth=1.5, alpha=0.8)
    
    # Format plot
    ax.set_xlabel('Time Windows', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'Entropy Across All Files ({len(all_files)} files)', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside plot
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot:
        norm_str = 'normalized' if normalize else 'raw'
        save_path = os.path.join(folder_path, f'entropy_comparison_{norm_str}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved entropy plot to: {save_path}")
    
    # Close the figure to prevent display
    plt.close(fig)
    
    return fig, ax


def plot_all_speed(folder_path, normalize=True, save_plot=True):
    """
    Load all *_dysco.npy files from a folder and plot their reconfiguration speed on one plot.
    
    Parameters:
    -----------
    folder_path : str
        Path to the folder containing the *_dysco.npy files
    normalize : bool
        Whether to min-max normalize the data before plotting (default: True)
    save_plot : bool
        Whether to save the plot to the folder (default: True)
    """
    # Find all files ending with _dysco.npy
    all_files = [f for f in os.listdir(folder_path) if f.endswith('_dysco.npy')]
    
    if not all_files:
        print(f"No *_dysco.npy files found in {folder_path}")
        return
    
    print(f"Found {len(all_files)} files to plot for reconfiguration speed")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Use a colormap to generate different colors for each file
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_files)))
    
    # Plot each file
    for i, filename in enumerate(all_files):
        full_path = os.path.join(folder_path, filename)
        
        # Load data
        data = np.load(full_path, allow_pickle=True).item()
        
        # Get speed
        if "speed" in data:
            speed = data["speed"]
        else:
            print(f"Warning: No speed found in {filename}, skipping")
            continue
        
        # Normalize if requested
        if normalize:
            speed = (speed - np.min(speed)) / (np.max(speed) - np.min(speed))
            ylabel = 'Normalized Reconfiguration Speed'
        else:
            ylabel = 'Reconfiguration Speed'
        
        # Use filename without '_dysco' as label
        label = filename.replace('_dysco.npy', '')
        
        # Plot
        ax.plot(speed, color=colors[i], label=label, linewidth=1.5, alpha=0.8)
    
    # Format plot
    ax.set_xlabel('Time Windows', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'Reconfiguration Speed Across All Files ({len(all_files)} files)', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside plot
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot:
        norm_str = 'normalized' if normalize else 'raw'
        save_path = os.path.join(folder_path, f'speed_comparison_{norm_str}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved speed plot to: {save_path}")
    
    # Close the figure to prevent display
    plt.close(fig)
    
    return fig, ax


def plot_all_norm2(folder_path, normalize=True, save_plot=True):
    """
    Load all *_dysco.npy files from a folder and plot their norm2 on one plot.
    
    Parameters:
    -----------
    folder_path : str
        Path to the folder containing the *_dysco.npy files
    normalize : bool
        Whether to min-max normalize the data before plotting (default: True)
    save_plot : bool
        Whether to save the plot to the folder (default: True)
    """
    # Find all files ending with _dysco.npy
    all_files = [f for f in os.listdir(folder_path) if f.endswith('_dysco.npy')]
    
    if not all_files:
        print(f"No *_dysco.npy files found in {folder_path}")
        return
    
    print(f"Found {len(all_files)} files to plot for norm2")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Use a colormap to generate different colors for each file
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_files)))
    
    # Plot each file
    for i, filename in enumerate(all_files):
        full_path = os.path.join(folder_path, filename)
        
        # Load data
        data = np.load(full_path, allow_pickle=True).item()
        
        # Get norm2
        if "norm2" in data:
            norm2 = data["norm2"]
        else:
            print(f"Warning: No norm2 found in {filename}, skipping")
            continue
        
        # Normalize if requested
        if normalize:
            norm2 = (norm2 - np.min(norm2)) / (np.max(norm2) - np.min(norm2))
            ylabel = 'Normalized Norm-2'
        else:
            ylabel = 'Norm-2'
        
        # Use filename without '_dysco' as label
        label = filename.replace('_dysco.npy', '')
        
        # Plot
        ax.plot(norm2, color=colors[i], label=label, linewidth=1.5, alpha=0.8)
    
    # Format plot
    ax.set_xlabel('Time Windows', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'Norm-2 Across All Files ({len(all_files)} files)', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside plot
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot:
        norm_str = 'normalized' if normalize else 'raw'
        save_path = os.path.join(folder_path, f'norm2_comparison_{norm_str}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved norm2 plot to: {save_path}")
    
    # Close the figure to prevent display
    plt.close(fig)
    
    return fig, ax


def plot_all_measures(folder_path, normalize=True, save_plots=True):
    """
    Plot all three measures (entropy, speed, norm2) from files in a folder.
    
    Parameters:
    -----------
    folder_path : str
        Path to the folder containing the *_dysco.npy files
    normalize : bool
        Whether to min-max normalize the data before plotting (default: True)
    save_plots : bool
        Whether to save the plots to the folder (default: True)
    """
    print(f"\n{'='*60}")
    print(f"Plotting all measures for folder: {folder_path}")
    print('='*60)
    
    plot_all_entropy(folder_path, normalize=normalize, save_plot=save_plots)
    plot_all_speed(folder_path, normalize=normalize, save_plot=save_plots)
    plot_all_norm2(folder_path, normalize=normalize, save_plot=save_plots)
    
    print(f"\nAll plots saved to: {folder_path}")


# Example usage - just change the folder path
if __name__ == "__main__":
    # Turn off interactive mode completely
    plt.ioff()
    
    # List of participant folders to process
    participant_folders = [
        "C:/Users/naima/DySCo-main/DySCo-main/DELFT_NEW/P002_dysco_output/",
        "C:/Users/naima/DySCo-main/DySCo-main/DELFT_NEW/P003_dysco_output/",
        "C:/Users/naima/DySCo-main/DySCo-main/DELFT_NEW/P004_dysco_output/",
        "C:/Users/naima/DySCo-main/DySCo-main/DELFT_NEW/P005_dysco_output/"
    ]
    
    # Plot for each participant folder
    for folder_path in participant_folders:
        if os.path.exists(folder_path):
            plot_all_measures(folder_path, normalize=True, save_plots=True)
        else:
            print(f"Folder not found: {folder_path}")
    
    print("\nAll plotting complete! No windows were displayed.")