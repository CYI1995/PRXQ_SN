import numpy as np
import math
import time
import os
from multiprocessing import Pool, cpu_count


# ==================== Parameters ====================
d = 16
Samples = 10000
N_list = np.array([3,5,7,9])
k_list = np.array([1,2,3,4,5,6,7])
RandomVecs = np.load('RandomVectors_d16.npy')
MMS = np.identity(d*d)/(d*d)

def partial_trace_fast(rho):
    return np.einsum('jiki->jk', rho.reshape(d, d, d, d))

def hankel_matrix_fast(mat,N):

    L_hankel = int((N+1)/2)
    ev = np.linalg.eigvalsh(mat)
    powers = np.arange(1, 17)[:, None]
    moments = np.sum(ev**powers, axis=1).real
    
    H = np.zeros((L_hankel, L_hankel))
    for i in range(L_hankel):
        for j in range(i, L_hankel):
            idx = i + j
            val = moments[idx]
            H[i, j] = H[j, i] = val

    return H


def check_momentRM(Rkrho, N):
    H = hankel_matrix_fast(Rkrho, N)
    eig = np.linalg.eigvalsh(H)
    min_eig = np.min(eig).real
    return 1 if min_eig < -1e-12 else 0

def process_sample(n):
    vec = RandomVecs[n]
    rho = 0.5 * np.outer(vec, vec.conj()) + 0.5 * MMS
    
    # Initialize with the correct shape (k_list=7, N_list=4)
    # Using float32 here prevents casting issues later
    res_RM = np.zeros((len(k_list), len(N_list)))

    rho_A = partial_trace_fast(rho)
    # Optimization: pre-calculate the identity matrix
    eye_d = np.eye(d, dtype=rho_A.dtype)
    kron_base = np.kron(rho_A, eye_d)

    for col_idx, N in enumerate(N_list):
        for row_idx, k in enumerate(k_list):
            Rkrho = k * kron_base - rho
            res_RM[row_idx, col_idx] = check_momentRM(Rkrho, N)
        
    return res_RM


def main():
    # 检查文件
    
    # 初始化总结果
    total_RM = np.zeros((len(k_list), len(N_list)))

    with Pool(cpu_count()) as pool:
        # 提交所有任务
        async_results = []
        for i in range(Samples):
            async_results.append(pool.apply_async(process_sample, (i,)))
        
        # 获取结果并显示进度
        results = []
        for i, res in enumerate(async_results):
            results.append(res.get())
            if i % 200 == 0:
                print(f"  Completed {i+1}/{Samples} samples")
        
        # 继续使用results...
    
    # 汇总结果
    for sample_RM in results:
        total_RM += sample_RM
    

    total_RM /= Samples
    

    np.save('RMd16.npy', total_RM)
    

# ==================== 运行 ====================
if __name__ == '__main__':
    # 防止多线程冲突
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    main()