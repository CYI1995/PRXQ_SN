import numpy as np
import os
from multiprocessing import Pool, cpu_count

# ==================== 参数 ====================
klist = np.array([2, 3, 4, 5, 6, 7, 8, 9])
Samples = 50000
TOTAL_DIM = 256  # 16 * 16
d = 16
L = 8 
N = 16

# ==================== 核心函数 ====================
def partial_trace(rho):
    return np.einsum('jiki->jk', rho.reshape(d, d, d, d))

def process_sample(sample_idx):
    """处理单个样本"""
    # 初始化这个样本的结果
    sample_RM = np.zeros((9, 5), dtype=float)
    
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
        
        # 计算偏迹
        rho_A = partial_trace(rho)
        I_B = np.identity(16, dtype=complex)
        kron_base = np.kron(rho_A, I_B)
        
        # 对每个k值计算
        for k_idx, k in enumerate(klist):
            Rkrho = k * kron_base - rho
            eigvals_H = np.linalg.eigvals(Rkrho)
            if np.min(eigvals_H.real) < -1e-12:
                sample_RM[k_idx, col] = 1.0
        
        del data
    
    return sample_RM

# ==================== 主函数 ====================
def main():
    # 检查文件
    files = ['RMSK2.npy', 'RMSK3.npy', 'RMSK4.npy', 'RMSK5.npy', 'RMSK6.npy']
    for f in files:
        if not os.path.exists(f):
            print(f"Error: {f} not found!")
            return 
    
    # 初始化总结果
    total_RM = np.zeros((9, 5), dtype=float)
    
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
    

    np.save('Data_Table2.npy', total_RM)
    

# ==================== 运行 ====================
if __name__ == '__main__':
    # 防止多线程冲突
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    main()