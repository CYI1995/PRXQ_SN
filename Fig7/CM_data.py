import numpy  as np
import scipy
import math
from scipy import linalg
import matplotlib
from matplotlib import pyplot as plt
import source as mycode

import numpy as np
from itertools import combinations



def gell_mann_matrices(d: int, convention: str = "physics") -> list[np.ndarray]:
    """
    Return the d²−1 traceless generalized Gell-Mann matrices for SU(d).
    These are the standard generators of the special unitary group SU(d).
    
    Parameters
    ----------
    d : int
        Dimension (d=2 → Pauli matrices, d=3 → λ-matrices of QCD, d=4 → 15 matrices, etc.)
    convention : str
        - "physics" : Tr(λ_a λ_b) = 2 δ_{ab}   (most common in particle physics and QI)
        - "quantum_info" : sometimes people use Tr(λ_a λ_b) = δ_{ab} (divided by √2)
    
    Returns
    -------
    list of (d, d) complex np.ndarrays
        The d Gell-Mann matrices λ_1, ..., λ_{d²−1}
    """
    if d < 2:
        raise ValueError("d must be >= 2")
    
    matrices = []

    matrices.append(np.identity(d)/np.sqrt(d))
    factor = 1.0
    if convention == "quantum_info":
        factor = 1 / np.sqrt(2)   # makes Tr(λ_a λ_b) = δ_{ab}
    # Default "physics" convention keeps Tr(λ_a λ_b) = 2 δ_{ab}
    
    # 1. Symmetric off-diagonal: |i⟩⟨j| + |j⟩⟨i|   (i < j)
    for i, j in combinations(range(d), 2):
        mat = np.zeros((d, d), dtype=complex)
        mat[i, j] = mat[j, i] = 1.0
        matrices.append(mat/np.sqrt(2))
    
    # 2. Antisymmetric off-diagonal: -i (|i⟩⟨j| − |j⟩⟨i|)   (i < j)
    for i, j in combinations(range(d), 2):
        mat = np.zeros((d, d), dtype=complex)
        mat[i, j] = -1j
        mat[j, i] = 1j
        matrices.append(mat/np.sqrt(2))
    
    # 3. Diagonal matrices (d−1 of them)
    for k in range(1, d):
        mat = np.zeros((d, d), dtype=complex)
        for m in range(k):
            mat[m, m] = 1.0
        mat[k, k] = -k
        
        # Normalization for the standard physics convention Tr(λ_a λ_b)=2δ_{ab}
        norm = np.sqrt(2.0 / (k * (k + 1.0)))
        matrices.append(norm * mat/np.sqrt(2))
    
    return matrices

def partial_trace(rho, dA, dB):

    D = len(rho[0])
    rhoA = np.zeros((dA,dA),dtype = complex)

    if(dA * dB == D):
        for i in range(dA):
            for j in range(dA):
                entry = 0 + 1j*0
                for x in range(dB):
                    idx1 = i * dB + x
                    idx2 = j * dB + x 
                    entry = entry + rho[idx1][idx2]

                rhoA[i][j] = entry 

        return rhoA 
    
    else:

        print('Error')

        return 0
    

def RandomState(eps,d):

    U = mycode.RandomUnitary(d*d)
    v = U[0,:]

    return (1-eps) * np.outer(v, v.conj()) + eps * np.identity(d*d)/(d*d) 

def Hankel(mat,N,k):

    L = int((N+1)/2)

    dim = len(mat[0])
    mat_power = np.identity(dim)
    moment_list = np.zeros(N)

    for n in range(N):
        mat_power = mat_power.dot(mat)
        moment_list[n] = mycode.trace(mat_power).real 


    H = np.zeros((L,L))
    if(N % 2 == 1):
        
        for i in range(L):
            for j in range(i,L):
                H[i][j] = moment_list[i+j]
                H[j][i] = H[i][j]
        return H 
    
    else: 
        for i in range(L):
            for j in range(i,L):
                H[i][j] = k * moment_list[i+j] - moment_list[i+j+1]
                H[j][i] = H[i][j]
        return H 



def moment_based_k_reduction_criterion(rho,k,d,N):

    #If Hankel(Rk(rho)) \ge 0, return 0; else return 1.

    I_B = np.identity(d)

    if(len(rho[0]) == d*d):

        rho_A = partial_trace(rho, d, d)
        Rkrho = k*np.kron(rho_A,I_B) - rho 
        H = Hankel(Rkrho,N,k)
        eig,vec = np.linalg.eig(H)
        if(np.min(eig) < 0):

            return 1
        else:
            return 0 

    else:

        print('Error')

        return -1
    

def moment_based_CM(T_cor,k,d):


    U, S, Vh = np.linalg.svd(T_cor, full_matrices=True)
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
        

SampleNum = 1000
samples = np.random.dirichlet((1,1,1,1,1,1), size=SampleNum)


k = 5
num = 16
Listofd = np.zeros(num)
ListofRatio_CM = np.zeros(num)


for nd in range(num):

    d = 7 + nd

    basis = gell_mann_matrices(d, convention="physics")
    UL = np.zeros((d*d,d*d),dtype=complex)
    for i1 in range(d):
        for j1 in range(d):
            idx1 = i1*d + j1
            for idx2 in range(len(basis)):
                opt_temp = basis[idx2]
                UL[idx2][idx1] = opt_temp[j1][i1] 

    NumberofPassing_CM = 0

    for n in range(SampleNum):

        print(d,n)

        coefficients = samples[n]
        vec_lambda = np.zeros(d*d)
        for i in range(6):
            for j in range(6):
                idx = 6*i + j
                vec_lambda[idx] = math.sqrt(coefficients[i] * coefficients[j])
        Mat_lambda = np.diag(vec_lambda)

        T_cor = (UL).dot(Mat_lambda.dot(UL.T))
        for col in range(d*d):
            T_cor[0][col] = 0
            T_cor[col][0] = 0

        NumberofPassing_CM = NumberofPassing_CM + moment_based_CM(T_cor,k,d)

    Listofd[nd] = d 
    ListofRatio_CM[nd] = NumberofPassing_CM/SampleNum 

np.save('CM.npy',ListofRatio_CM)
plt.scatter(Listofd,ListofRatio_CM)
plt.ylim(0,0.7)
plt.show()





