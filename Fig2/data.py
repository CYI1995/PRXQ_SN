import numpy  as np
import scipy
import math
from scipy import linalg
import matplotlib
from matplotlib import pyplot as plt
import source as mycode
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
    factor = 1.0
    if convention == "quantum_info":
        factor = 1 / np.sqrt(2)   # makes Tr(λ_a λ_b) = δ_{ab}
    # Default "physics" convention keeps Tr(λ_a λ_b) = 2 δ_{ab}
    
    # 1. Symmetric off-diagonal: |i⟩⟨j| + |j⟩⟨i|   (i < j)
    for i, j in combinations(range(d), 2):
        mat = np.zeros((d, d), dtype=complex)
        mat[i, j] = mat[j, i] = 1.0
        matrices.append(mat)
    
    # 2. Antisymmetric off-diagonal: -i (|i⟩⟨j| − |j⟩⟨i|)   (i < j)
    for i, j in combinations(range(d), 2):
        mat = np.zeros((d, d), dtype=complex)
        mat[i, j] = -1j
        mat[j, i] = 1j
        matrices.append(mat)
    
    # 3. Diagonal matrices (d−1 of them)
    for k in range(1, d):
        mat = np.zeros((d, d), dtype=complex)
        for m in range(k):
            mat[m, m] = 1.0
        mat[k, k] = -k
        
        # Normalization for the standard physics convention Tr(λ_a λ_b)=2δ_{ab}
        norm = np.sqrt(2.0 / (k * (k + 1.0)))
        matrices.append(norm * mat)
    
    # Apply global convention factor if needed
    if convention == "quantum_info":
        matrices = [M / np.sqrt(2) for M in matrices]
    
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


def k_reduction_criterion(rho,k):

    #If Rk(rho) \ge 0, return 0; else return 1.

    I_B = np.identity(8)

    if(len(rho[0]) == 64):

        rho_A = partial_trace(rho, 8, 8)
        Rkrho = k*np.kron(rho_A,I_B) - rho 
        eig,vec = np.linalg.eig(Rkrho)
        if(np.min(eig) < 0):
            return 1
        else:
            return 0 
        

    else:

        print('Error')

        return -1

def CMonenorm(rho):

    d = int(math.sqrt(len(rho[0])))

    basis = gell_mann_matrices(d, convention="physics")
    L = len(basis)

    T_cor = np.zeros((L,L),dtype = complex)
    for i in range(L):
        P1 = basis[i]*math.sqrt(1/2)
        for j in range(L):
            P2 = basis[j]*math.sqrt(1/2)
            T_cor[i][j] = mycode.trace(rho.dot(np.kron(P1,P2))).real

    U, S, Vh = np.linalg.svd(T_cor, full_matrices=True)
    
    return np.sum(S)


r = 4
Numd = 16

Listofd = np.zeros(Numd)
ListofRM = np.zeros(Numd)
ListofCM = np.zeros(Numd)

for n in range(Numd):

    d = 4*n + 4 

    print(d)

    vec = np.zeros(d*d) 
    for i in range(r):
        vec[(d+1)*i] = 1/math.sqrt(r)

    rho = np.outer(vec,vec.conj())
    # T1 = CMonenorm(rho)
    T1 = 4 - 1/d

    Listofd[n] = d 
    ListofCM[n] = 1 - (r-1-1/d)/T1 
    ListofRM[n] = 1/(1 + (r*r-r)/d - r/(d*d))

np.save('ListofCM.npy',ListofCM)
np.save('ListofRM.npy',ListofRM)

plt.plot(Listofd,ListofCM)
plt.plot(Listofd,ListofRM)
plt.show()



