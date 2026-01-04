import numpy as np
import math
import time
import os
from multiprocessing import Pool, cpu_count
from itertools import combinations
import matplotlib.pyplot as plt

# ==================== Parameters ====================
SampleNum = 10000
k_value = 5
d_min, d_max = 7, 22
num_d = d_max - d_min + 1
d_range = range(d_min, d_max + 1)

# Load shared data
samples = np.load('RandomDirichlet.npy')  # Expected shape (SampleNum, >=36)

# ==================== Helper Functions ====================

def get_gell_mann_basis(d):
    """Cached version to avoid recomputing basis for same d."""
    matrices = []
    
    # Identity
    matrices.append(np.identity(d) / np.sqrt(d))
    
    # Symmetric off-diagonal (d(d-1)/2 matrices)
    for i, j in combinations(range(d), 2):
        mat = np.zeros((d, d), dtype=complex)
        mat[i, j] = mat[j, i] = 1.0 / np.sqrt(2)
        matrices.append(mat)
    
    # Antisymmetric off-diagonal (d(d-1)/2 matrices)
    for i, j in combinations(range(d), 2):
        mat = np.zeros((d, d), dtype=complex)
        mat[i, j], mat[j, i] = -1j / np.sqrt(2), 1j / np.sqrt(2)
        matrices.append(mat)

    for k in range(1, d):
        mat = np.zeros((d, d), dtype=complex)
        for m in range(k):
            mat[m, m] = 1.0
        mat[k, k] = -k
        norm = np.sqrt(2.0 / (k * (k + 1.0)))
        matrices.append((norm * mat / np.sqrt(2)))
    
    return np.array(matrices)

def compute_T(rho,d,Ps):

    rho_tensor = rho.reshape((d, d, d, d))
    T_block = np.einsum('abcd, ica, jdb -> ij', rho_tensor, Ps, Ps, optimize=True)
    
    T_block = T_block.real
    T_block[0, :] = 0.0
    T_block[:, 0] = 0.0
    return T_block


def hankel_fast(mat, N, k):
    L = int((N + 1) / 2)

    ev = np.linalg.eigvalsh(mat)
    powers = np.arange(1, N + 1)[:, None]
    moment_list = np.sum(ev**powers, axis=1).real

    H = np.zeros((L, L))
    if N % 2 == 1:
        for i in range(L):
            for j in range(i, L):
                H[i, j] = H[j, i] = moment_list[i + j]
    else:
        for i in range(L):
            for j in range(i, L):
                H[i, j] = H[j, i] = k * moment_list[i + j] - moment_list[i + j + 1]
    return H

def moment_based_CM(T, k, d):
    S = np.linalg.svd(T, compute_uv=False)
    T2, T4 = np.sum(S**2), np.sum(S**4)
    S2 = (d / (d - 1))**2 * T2 
    S4 = (2/3) * (d / (d - 1))**4 * T4 + (S2**2) / 3
    Bk = (d * k - 1) / (d - 1)
    n_val = int(Bk**2 / S2) if S2 > 0 else 0

    if n_val == 0: return 1
    if n_val >= d**2 - 1:
        fn = (d**2 + 1) * S2**2 / (3 * (d**2 - 1))
        return 1 if S4 < fn else 0
    else:
        gn = n_val * (n_val + 1) * S2 - n_val * Bk**2 
        sqgn = math.sqrt(max(0, gn))
        fn = ((sqgn - Bk)**4 + (sqgn + n_val * Bk)**4 / n_val**3) * 2 / (3 * (n_val + 1)**4) + (S2**2) / 3 
        return 1 if S4 < fn else 0

# ==================== Worker Logic ====================

# Global cache for basis matrices to be initialized once per process
BASIS_CACHE = {}

def process_sample(n):
    coeffs = samples[n]
    # Results: [RM3, RM4, RM5, RM6, RM7, RM8, CM] for each d
    sample_results = np.zeros((num_d, 7))
    
    for idx, d in enumerate(d_range):

        basis = get_gell_mann_basis(d)
        # 2. RM Path: Reconstruct Rho
        v = np.zeros(d*d, dtype=complex)
        for i in range(6):
            v[i*(d+1)] = math.sqrt(coeffs[i])
        rho = np.outer(v, v.conj())
        
        # RM Criteria N=3 to 8
        rho_A = np.einsum('jiki->jk', rho.reshape(d, d, d, d))
        Rkrho = k_value * np.kron(rho_A, np.eye(d, dtype=complex)) - rho
        for N_idx, N in enumerate(range(3, 9)):
            H = hankel_fast(Rkrho, N, k_value)
            min_eig = np.linalg.eigvalsh(H)[0]
            sample_results[idx, N_idx] = 1 if min_eig < -1e-9 else 0
            
        # 3. CM Path: Construct T_cor

        T_cor = compute_T(rho, d, basis)
        sample_results[idx, 6] = moment_based_CM(T_cor.real, k_value, d)
            
    return sample_results

# ==================== Main ====================

def main():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    print(f"Starting merge computation for {SampleNum} samples (d={d_min}-{d_max})...")
    start_time = time.time()
    
    with Pool(max(1, cpu_count() - 1)) as pool:
        chunk = max(1, SampleNum // (cpu_count() * 2))
        raw_results = list(pool.imap(process_sample, range(SampleNum), chunksize=chunk))

    # Aggregation
    final_matrix = np.mean(raw_results, axis=0) # (num_d, 7)
    
    # Save RM results
    labels = ['RM3', 'RM4', 'RM5', 'RM6', 'RM7', 'RM8']
    for i, label in enumerate(labels):
        np.save(f'{label}.npy', final_matrix[:, i])
    
    # Save CM results
    np.save('CM.npy', final_matrix[:, 6])
    
    # Plotting
    Listofd = np.array(list(d_range))
    plt.figure(figsize=(10, 6))
    for i, label in enumerate(labels):
        plt.plot(Listofd, final_matrix[:, i], label=label)
    plt.scatter(Listofd, final_matrix[:, 6], label='CM', color='black', zorder=5)
    
    plt.xlabel('Dimension d')
    plt.ylabel('Violation Ratio')
    plt.title(f'Comparison of Criteria (k={k_value})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('Comparison_Plot.png')
    print(f"Done in {time.time() - start_time:.1f}s. Results saved to RM3-8.npy and CM.npy")

if __name__ == '__main__':
    main()