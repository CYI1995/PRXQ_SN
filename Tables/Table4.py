import numpy  as np
import scipy
import math
from scipy import linalg
import matplotlib
from matplotlib import pyplot as plt


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

def k_reduction_criterion(rho,k):

    #If Rk(rho) \ge 0, return 0; else return 1.

    I_B = np.identity(16)
    rho_A = partial_trace(rho, 16, 16)
    Rkrho = k*np.kron(rho_A,I_B) - rho 
    eig,vec = np.linalg.eig(Rkrho)
    if(np.min(eig) < 0):
        return 1
    else:
        return 0 


def Hankel(mat,N,k):

    L = int((N+1)/2)

    dim = len(mat[0])
    mat_power = np.identity(dim)
    moment_list = np.zeros(N)

    for n in range(N):
        mat_power = mat_power.dot(mat)
        moment_list[n] = np.trace(mat_power).real 

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

d = 16
D = d*d
I_B = np.identity(d)
k_list = np.array([3,4,5,6,7,8,9,10])
N_list = np.array([6,7,8,9])

for i in range(8):
    k = k_list[i]
    MES = np.zeros(D)
    for j in range(k):
        MES[(d+1)*j] = 1/math.sqrt(k)
    BELL = np.zeros(D)
    BELL[(k-1)*d + k] = 1/math.sqrt(2)
    BELL[k*d+k-1] = 1/math.sqrt(2) 

    rho = 0.5*np.outer(MES,MES.conj()) + 0.5*np.outer(BELL,BELL.conj())
    rhoA = partial_trace(rho, d, d)
    Rkrho = (k-1)*np.kron(rhoA,I_B) - rho 

    minimal_order = 0
    for l in range(4):
        N = N_list[l]
        H = Hankel(Rkrho,N,k-1)
        eig = np.linalg.eigvals(H)
        if(np.min(eig).real < 0):
            minimal_order = N 
            break 

    eigs = np.linalg.eigvals(Rkrho)
    print(k, (-1)*np.min(eigs).real, minimal_order)






