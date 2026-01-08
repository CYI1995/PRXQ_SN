import numpy as np

def partial_trace(rho, dim_A, dim_B):
    """Vectorized partial trace over subsystem B (dim_B)"""
    rho_reshaped = rho.reshape((dim_A, dim_B, dim_A, dim_B))
    return np.trace(rho_reshaped, axis1=1, axis2=3)

d = 16
dim_A = d * d          # 256
Samples = 50000
Ks = [2, 3, 4, 5, 6]

rng = np.random.default_rng()  # 更现代的随机数生成器

for K in Ks:
    dim_B = K
    dim_total = dim_A * dim_B
    RandomStates = []
    
    print(f"Starting K={K} (dim_total={dim_total}), generating {Samples} states...")
    
    for n in range(Samples):
        # 生成复高斯随机向量并归一化
        z = rng.normal(0, 1, dim_total) + 1j * rng.normal(0, 1, dim_total)
        z /= np.linalg.norm(z)          # 归一化得到 |ψ⟩
        
        # 纯态投影算符 |ψ⟩⟨ψ|
        psi = np.outer(z, z.conj())
        
        # partial trace over B，得到 subsystem A 的约化密度矩阵
        rho_A = partial_trace(psi, dim_A, dim_B)
        
        RandomStates.append(rho_A)
        
        # 可选：每2000个样本打印一次进度（避免太频繁IO）
        if (n + 1) % 2000 == 0:
            print(f"  K={K}: {n+1}/{Samples} completed")
    
    # 保存为 .npy 文件
    np.save(f'RMSK{K}.npy', np.array(RandomStates))  # 建议转成array再保存，便于后续加载
    print(f"K={K} finished and saved to RMSK{K}.npy\n")
