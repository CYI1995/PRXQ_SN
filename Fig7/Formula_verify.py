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
        The d²−1 Gell-Mann matrices λ_1, ..., λ_{d²−1}
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

def CM(rho,basis):

    d = int(math.sqrt(len(rho[0])))
    L = len(basis)

    T_cor = np.zeros((L,L),dtype = complex)
    for i in range(L):
        P1 = basis[i]
        for j in range(L):
            P2 = basis[j]
            T_cor[i][j] = mycode.trace(rho.dot(np.kron(P1,P2)))

    return T_cor
        

SampleNum = 100
samples = np.random.dirichlet((1,1,1,1,1,1), size=SampleNum)

d = 6
basis = gell_mann_matrices(d, convention="physics")

coefficients = samples[0]
state = np.zeros(d*d)
for i in range(len(coefficients)):
    state[i*(d+1)] = math.sqrt(coefficients[i])
rho = np.outer(state, state.conj())

T = CM(rho,basis)


idx1 = 1
idx2 = 2
P1 = basis[idx1]
P2 = basis[idx2]


temp = 0
for i in range(d):
    psii = math.sqrt(coefficients[i])
    for j in range(d):
        psij = math.sqrt(coefficients[j])
        temp = temp + psii*psij*P1[j][i]*P2[j][i]

vec_lambda = np.zeros(d*d)
for i in range(d):
    for j in range(d):
        idx = d*i + j
        vec_lambda[idx] = math.sqrt(coefficients[i] * coefficients[j])
Mat_lambda = np.diag(vec_lambda)

UL = np.zeros((d*d,d*d),dtype=complex)
for i1 in range(d):
    for j1 in range(d):
        idx1 = i1*d + j1
        for idx2 in range(len(basis)):
            opt_temp = basis[idx2]
            UL[idx2][idx1] = opt_temp[j1][i1] 

Td = (UL).dot(Mat_lambda.dot(UL.T))
T_diff = Td - T 

print(mycode.matrix_norm(T_diff.dot(np.conj(T_diff))))



