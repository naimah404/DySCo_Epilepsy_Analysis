import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


def dysco_distance(matrix_a, matrix_b, what_distance):
    with np.errstate(invalid='ignore'):
        matrix_a = matrix_a.copy()
        matrix_b = matrix_b.copy()

        n_eigen = matrix_a.shape[1]

        # Define minimatrix
        minimatrix = np.zeros((2 * n_eigen, 2 * n_eigen))

        # Fill diagonal with the squared norms of eigenvectors
        for i in range(n_eigen):
            minimatrix[i, i] = np.dot(matrix_a[:, i].T, matrix_a[:, i])
            minimatrix[n_eigen + i, n_eigen + i] = -np.dot(matrix_b[:, i].T, matrix_b[:, i])

        # Fill the rest with scalar products
        minimatrix_up_right = np.dot(matrix_a.T, matrix_b)
        minimatrix[0:n_eigen, n_eigen:2 * n_eigen] = minimatrix_up_right
        minimatrix[n_eigen:2 * n_eigen, 0:n_eigen] = -minimatrix_up_right.T

        # Compute eigenvalues
        if what_distance != 2:
            lambdas = np.linalg.eigvals(minimatrix)
            lambdas = np.real(lambdas)

        if what_distance == 1:
            distance = np.sum(np.abs(lambdas))
        elif what_distance == 2:
            # Modify the distance calculation
            distance = np.sqrt(np.sum(np.diag(minimatrix) ** 2) - 2 * np.sum(minimatrix_up_right ** 2))
        else:
            distance = np.max(lambdas)

    return distance


def compute_fcd_matrix(eigenvectors, n_eigs_to_use=8):
    """
    Compute FCD matrix and reconfiguration matrix in PARALLEL.
    Parameters:
    -----------
    eigenvectors : np.ndarray
        Eigenvectors of shape (T, n_regions, n_eigen)
    n_eigs_to_use : int
        Number of eigenvectors to use (default: 8)
    Returns:
    --------
    fcd : np.ndarray
        FCD matrix (T x T)
    fcd_reconf : np.ndarray
        Reconfiguration matrix (T x T)
    """
    T = eigenvectors.shape[0]
    n_eigs_actual = min(n_eigs_to_use, eigenvectors.shape[2])
    eigvect = eigenvectors[:, :, :n_eigs_actual]
    fcd = np.zeros((T, T))
    fcd_reconf = np.zeros((T, T))
    
    print(" Computing FCD and reconfiguration matrices in parallel...")
    # Parallel computation of all i,j pairs
    results = Parallel(n_jobs=-1)(
        delayed(_compute_fcd_single)(i, j, eigvect, 2)  # Pass distance type directly
        for i in range(T) for j in range(i, T)
    )
    total_iterations = len(results)
    progress_bar = tqdm(total=total_iterations, desc=" Filling matrices")
    for i, j, fcd_ij, fcd_reconf_ij in results:
        fcd[i, j] = fcd_ij
        fcd[j, i] = fcd_ij
        fcd_reconf[i, j] = fcd_reconf_ij
        fcd_reconf[j, i] = fcd_reconf_ij
        progress_bar.update(1)
    progress_bar.close()
    return fcd, fcd_reconf


def _compute_fcd_single(i, j, eigvect, what_distance):
    """
    Helper function to compute FCD for a single pair of timepoints.
    Used for parallel processing.
    """
    matrix_a = eigvect[i, :, :]
    matrix_b = eigvect[j, :, :]
    fcd_ij = dysco_distance(matrix_a, matrix_b, what_distance)
    fcd_reconf_ij = 0  # Placeholder - implement if needed
    return i, j, fcd_ij, fcd_reconf_ij