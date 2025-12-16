import numpy  as np
import scipy
import math
from scipy import linalg
import matplotlib
from matplotlib import pyplot as plt
import source as mycode

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

    M = d*d  # or whatever dimension
    z = np.random.normal(0, 1, M) + 1j * np.random.normal(0, 1, M)
    v = z / np.linalg.norm(z)

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

def correlationmatrx_criterion(T_cor,k):

    #If ||T||1 \le k-1/d, return 0; else return 1

    U, S, Vh = np.linalg.svd(T_cor, full_matrices=True)
    if(np.sum(S) > k-1/8):
        return 1 
    else:
        return 0


d = 8

Paulis = np.load('Pauli_basis.npy')
Samples = 1000
Ratiolist_RM0 = np.zeros(7)
Ratiolist_RM1 = np.zeros(7)
Ratiolist_RM2 = np.zeros(7)
Ratiolist_CM0 = np.zeros(7)
Ratiolist_CM1 = np.zeros(7)
Ratiolist_CM2 = np.zeros(7)
klist = np.array([1,2,3,4,5,6,7])
L = len(Paulis)


for n in range(Samples):

    print(n)

    rho_temp = RandomState(0,d)
    
    T_cor = np.zeros((L,L),dtype = complex)
    for i in range(1,L):
        P1 = Paulis[i]
        for j in range(1,L):
            P2 = Paulis[j]
            T_cor[i][j] = np.trace(rho_temp.dot(np.kron(P1,P2))).real/8

    for i in range(7):
        k = klist[i]
        if(k_reduction_criterion(rho_temp,k) == 0):
            break 
        else:
            Ratiolist_RM0[i] = Ratiolist_RM0[i] + 1

    for i in range(7):
        k = klist[i]
        if(k_reduction_criterion(0.9*rho_temp + 0.1*np.identity(d*d)/(d*d),k) == 0):
            break 
        else:
            Ratiolist_RM1[i] = Ratiolist_RM1[i] + 1

    for i in range(7):
        k = klist[i]
        if(k_reduction_criterion(0.5*rho_temp + 0.5*np.identity(d*d)/(d*d),k) == 0):
            break 
        else:
            Ratiolist_RM2[i] = Ratiolist_RM2[i] + 1

    for i in range(7):
        k = klist[i]
        if(correlationmatrx_criterion(T_cor,k) == 0):
            break 
        else:
            Ratiolist_CM0[i] = Ratiolist_CM0[i] + 1
        
    for i in range(7):
        k = klist[i]
        if(correlationmatrx_criterion(0.9*T_cor,k) == 0):
            break 
        else:
            Ratiolist_CM1[i] = Ratiolist_CM1[i] + 1

    for i in range(7):
        k = klist[i]
        if(correlationmatrx_criterion(0.5*T_cor,k) == 0):
            break 
        else:
            Ratiolist_CM2[i] = Ratiolist_CM2[i] + 1

Ratiolist_CM0 = Ratiolist_CM0/Samples
Ratiolist_RM0 = Ratiolist_RM0/Samples
Ratiolist_CM1 = Ratiolist_CM1/Samples
Ratiolist_RM1 = Ratiolist_RM1/Samples
Ratiolist_CM2 = Ratiolist_CM2/Samples
Ratiolist_RM2 = Ratiolist_RM2/Samples

np.save('Ratiolist_CM_eps0.npy',Ratiolist_CM0)
np.save('Ratiolist_RM_eps0.npy',Ratiolist_RM0)
np.save('Ratiolist_CM_eps01.npy',Ratiolist_CM1)
np.save('Ratiolist_RM_eps01.npy',Ratiolist_RM1)
np.save('Ratiolist_CM_eps05.npy',Ratiolist_CM2)
np.save('Ratiolist_RM_eps05.npy',Ratiolist_RM2)



