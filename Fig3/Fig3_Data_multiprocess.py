import numpy as np
import os
import time
from multiprocessing import Pool, cpu_count

# ==================== Parameters ====================
d = 8
Samples = 10000
klist = np.arange(1, 8)  # [1, 2, 3, 4, 5, 6, 7]
identity_64 = np.identity(64) / 64.0

# Load shared data
# Note: Ensure these files exist in the same directory
Paulis = np.load('Pauli_basis.npy') # Expected shape (64, 8, 8)
Ps = Paulis[1:]                     # Traceless parts (63, 8, 8)
RandomVectors = np.load('RandomVectors_d8.npy')

# ==================== Optimized Helper Functions ====================

def partial_trace_fast(rho):
    """Partial trace over system B using einsum."""
    return np.einsum('jiki->jk', rho.reshape(8, 8, 8, 8))

def k_reduction_criterion(rho, k):
    """Check k-reduction criterion: returns 1 if violated, 0 otherwise."""
    rho_A = partial_trace_fast(rho)
    # Rkrho = k * (rho_A \otimes I) - rho
    Rkrho = k * np.kron(rho_A, np.eye(8)) - rho
    # Using eigvalsh for Hermitian matrices is much faster than np.linalg.eig
    min_eig = np.linalg.eigvalsh(Rkrho)[0]
    return 1 if min_eig < -1e-9 else 0

def correlation_matrix_criterion(S, k):
    """Check CM criterion using pre-computed singular values S."""
    return 1 if np.sum(S) > (k - 1/8.0) + 1e-9 else 0

def compute_T(rho):
    """Vectorized calculation of the Correlation Matrix T_cor."""
    # Reshape rho to (8, 8, 8, 8) to separate system A and B indices
    rho_reshaped = rho.reshape(d, d, d, d)
    # Trace(rho * (P1 \otimes P2)) = sum rho_{ac, bd} * (P1)_{ba} * (P2)_{dc}
    T = np.einsum('acbd,iab,jcd->ij', rho_reshaped, Ps, Ps, optimize=True)
    return T.real / d

# ==================== Worker Logic ====================

def process_sample(n):
    """Process a single sample: returns results for all criteria and noise levels."""
    vec = RandomVectors[n]
    rho_pure = np.outer(vec, vec.conj())
    
    # Pre-calculate T_cor and its singular values for the pure state
    T_pure = compute_T(rho_pure)
    S_pure = np.linalg.svd(T_pure, compute_uv=False)
    
    # Results containers (Noise levels: 0.0, 0.1, 0.5)
    # index 0: RM, index 1: CM
    res_0 = np.zeros((2, 7)) 
    res_01 = np.zeros((2, 7))
    res_05 = np.zeros((2, 7))
    
    # Define noise mixtures for RM
    rho_01 = 0.9 * rho_pure + 0.1 * identity_64
    rho_05 = 0.5 * rho_pure + 0.5 * identity_64
    
    # For CM, mixing with identity (identity is orthogonal to Ps) 
    # effectively scales the S-values of the traceless T-matrix.
    S_01 = 0.9 * S_pure
    S_05 = 0.5 * S_pure

    for i, k in enumerate(klist):
        # Noise level 0.0
        res_0[0, i] = k_reduction_criterion(rho_pure, k)
        res_0[1, i] = correlation_matrix_criterion(S_pure, k)
        
        # Noise level 0.1
        res_01[0, i] = k_reduction_criterion(rho_01, k)
        res_01[1, i] = correlation_matrix_criterion(S_01, k)
        
        # Noise level 0.5
        res_05[0, i] = k_reduction_criterion(rho_05, k)
        res_05[1, i] = correlation_matrix_criterion(S_05, k)
        
    return res_0, res_01, res_05

# ==================== Main Execution ====================

def main():
    # Force single-thread for linear algebra within workers to prevent CPU thrashing
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    num_workers = max(1, cpu_count() - 1)
    print(f"Starting calculation with {num_workers} workers...")
    
    start_time = time.time()
    
    with Pool(num_workers) as pool:
        # Use imap to track progress
        results = list(pool.imap(process_sample, range(Samples), chunksize=10))
        
    # Aggregate results: Mean over samples to get Ratios
    # Final shapes will be (3 noise levels, 2 criteria, 7 k-values)
    final_0 = np.mean([r[0] for r in results], axis=0)
    final_01 = np.mean([r[1] for r in results], axis=0)
    final_05 = np.mean([r[2] for r in results], axis=0)
    
    # Save files
    np.save('Ratiolist_RM_eps0.npy', final_0[0])
    np.save('Ratiolist_CM_eps0.npy', final_0[1])
    
    np.save('Ratiolist_RM_eps01.npy', final_01[0])
    np.save('Ratiolist_CM_eps01.npy', final_01[1])
    
    np.save('Ratiolist_RM_eps05.npy', final_05[0])
    np.save('Ratiolist_CM_eps05.npy', final_05[1])
    
    print(f"✅ Finished in {time.time() - start_time:.2f} seconds.")
    print("Files saved: Ratiolist_RM/CM_eps0.npy, ...eps01.npy, ...05.npy")

if __name__ == '__main__':
    main()