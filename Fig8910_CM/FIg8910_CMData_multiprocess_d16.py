import numpy as np
import scipy
import math
from scipy import linalg
import matplotlib
from matplotlib import pyplot as plt
import source as mycode
from multiprocessing import Pool, cpu_count

# --- Keep your helper functions (partial_trace, Hankel, etc.) as they are ---
# (Included here by reference to your original code)


Ps = np.load('Pauli_basis_d16.npy')
d = 16
Samples = 10000
klist = np.array([1, 2, 3, 4, 5, 6, 7])
RandomVecs = np.load('RandomVectors_d16.npy')

def compute_T(rho):
    rho_tensor = rho.reshape((d, d, d, d))
    
    T_block = np.einsum('abcd, ica, jdb -> ij', rho_tensor, Ps, Ps, optimize=True)
    
    T_block = T_block.real / d
    T_block[0, :] = 0.0
    T_block[:, 0] = 0.0
    return T_block

def check_momentCM(S,k,d):

    #If the criterion is violated, return 1; else return 0. 

    T2 = 0
    T4 = 0
    for i in range(len(S)):
        T2 = T2 + S[i]**2 
        T4 = T4 + S[i]**4 

    S2 = (d/(d-1))**2 * T2 
    S4 = (2/3) * (d/(d-1))**4 * T4 + S2 * S2/3

    Bk = (d*k-1)/(d-1)
    n = int(Bk*Bk/(S2))

    if(n==0):
        return 1
    
    if(n>=d*d-1):

        fn = (d*d+1)*S2*S2/(3*(d*d-1))

        if(S4 < fn):
            return 1
        else:
            return 0

    else:
        gn = n*(n+1)*S2 - n*Bk*Bk 
        sqgn = math.sqrt(gn)
        fn = (sqgn - Bk)**4 + (sqgn + n*Bk)**4/n**3 
        fn = fn * 2 /(3 * (n+1)**4) 
        fn = fn + S2 * S2 /3 

        if(S4 < fn):
            return 1
        else:
            return 0
    


def process_sample(args):
    """
    Function to handle a single sample iteration.
    Returns a tuple of (results_0, results_1, results_2) for the k-lists.
    """
    v_temp, basis, klist, d = args
    L = len(basis)
    rho_temp = np.outer(v_temp, v_temp.conj())
    
    # Pre-calculate T_cor
    T_cor = compute_T(rho_temp)
    U, S, Vh = np.linalg.svd(T_cor, full_matrices=True)

    res0 = np.zeros(len(klist))
    res1 = np.zeros(len(klist))
    res2 = np.zeros(len(klist))

    # Scales to check
    scales = [1.0, 0.9, 0.5]
    res_refs = [res0, res1, res2]

    

    for scale, res_arr in zip(scales, res_refs):
        for i, k in enumerate(klist):
            if  check_momentCM(scale * S, k, d) == 0:
                break 
            else:
                res_arr[i] = 1
                
    return res0, res1, res2

if __name__ == '__main__':
    # Configuration



    # Prepare arguments for the worker pool
    tasks = [(RandomVecs[n], Ps, klist, d) for n in range(Samples)]

    print(f"Starting multiprocessing with {cpu_count()} cores...")
    
    # Execute in parallel
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_sample, tasks)

    # Aggregate results
    Ratiolist_CM0 = np.zeros(7)
    Ratiolist_CM1 = np.zeros(7)
    Ratiolist_CM2 = np.zeros(7)

    for r0, r1, r2 in results:
        Ratiolist_CM0 += r0
        Ratiolist_CM1 += r1
        Ratiolist_CM2 += r2

    # Normalize and Save
    Ratiolist_CM0 /= Samples
    Ratiolist_CM1 /= Samples
    Ratiolist_CM2 /= Samples

    np.save('Ratiolist_CM_d16e0.npy', Ratiolist_CM0)
    np.save('Ratiolist_CM_d16e01.npy', Ratiolist_CM1)
    np.save('Ratiolist_CM_d16e05.npy', Ratiolist_CM2)
    
    print("Processing complete. Files saved.")