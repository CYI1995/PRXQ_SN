import numpy as np
import os
from multiprocessing import Pool, cpu_count

# ==================== 参数 ====================
klist = np.array([4, 5, 6, 7, 8, 9])
Samples = 50000
TOTAL_DIM = 256  # 16 * 16
d = 16
L = 8 
N = 16

Paulis = np.load('Pauli_basis_d16.npy').astype(np.complex64)
Ps = Paulis[1:]  # (255, 16, 16)

# ==================== 核心函数 ====================
def compute_T(rho):
    """Vectorized calculation of the Correlation Matrix T_cor."""
    # Reshape rho to (8, 8, 8, 8) to separate system A and B indices
    rho_reshaped = rho.reshape(d, d, d, d)
    # Trace(rho * (P1 \otimes P2)) = sum rho_{ac, bd} * (P1)_{ba} * (P2)_{dc}
    T = np.einsum('acbd,iab,jcd->ij', rho_reshaped, Ps, Ps, optimize=True)
    return T.real / d

def process_sample(sample_idx):
    """处理单个样本"""
    # 初始化这个样本的结果
    sample_CM = np.zeros((9, 5), dtype=float)
    
    # 5个K-level文件
    files = ['RMSK2.npy', 'RMSK3.npy', 'RMSK4.npy', 'RMSK5.npy', 'RMSK6.npy']
    
    for col, fname in enumerate(files):
        if not os.path.exists(fname):
            continue
            
        # 加载数据
        data = np.load(fname, mmap_mode='r')
        if sample_idx >= data.shape[0]:
            del data
            continue
            
        rho = data[sample_idx]
        
        T = compute_T(rho)
        S = np.linalg.svd(T, compute_uv=False)
        
        # 对每个k值计算
        for k_idx, k in enumerate(klist):
            if np.sum(S) > k - 1/d + 1e-12:
                sample_CM[k_idx, col] = 1.0
        
        del data
    
    return sample_CM

# ==================== 主函数 ====================
def main():
    # 检查文件
    files = ['RMSK2.npy', 'RMSK3.npy', 'RMSK4.npy', 'RMSK5.npy', 'RMSK6.npy']
    for f in files:
        if not os.path.exists(f):
            print(f"Error: {f} not found!")
            return 
    
    # 初始化总结果
    total_CM = np.zeros((9, 5), dtype=float)
    
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
    for sample_CM in results:
        total_CM += sample_CM
    

    total_CM /= Samples
    

    np.save('Data_Table3.npy', total_CM)
    
    # 简单验证

# ==================== 运行 ====================
if __name__ == '__main__':
    # 防止多线程冲突
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    main()