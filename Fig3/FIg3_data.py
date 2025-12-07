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

def correlationmatrx_criterion(rho,k,Paulis):

    #If ||T||1 \le k-1/d, return 0; else return 1

    L = 63

    T_cor = np.zeros((L,L),dtype = complex)
    for i in range(1,L):
        P1 = Paulis[i]
        for j in range(1,L):
            P2 = Paulis[j]
            T_cor[i-1][j-1] = mycode.trace(rho.dot(np.kron(P1,P2)))/8


    U, S, Vh = np.linalg.svd(T_cor, full_matrices=True)
    if(np.sum(S) > k-1/8):
        return 1 
    else:
        return 0


d = 8

Paulis = np.load('Pauli_basis.npy')


Samples = 100
Ratiolist_RM = np.zeros(7)
Ratiolist_CM = np.zeros(7)
klist = np.array([1,2,3,4,5,6,7])


eps = 0.0

for k in range(1,d):

    print(k)

    NumberofPassing_CM = 0
    NumberofPassing_RM = 0

    for n in range(Samples):

        rho_temp = RandomState(eps,d)
        NumberofPassing_CM = NumberofPassing_CM + correlationmatrx_criterion(rho_temp,k,Paulis)
        NumberofPassing_RM = NumberofPassing_RM + k_reduction_criterion(rho_temp,k)

    Ratiolist_CM[k-1] = NumberofPassing_CM/Samples 
    Ratiolist_RM[k-1] = NumberofPassing_RM/Samples 


np.save('Ratiolist_CM_eps0.npy',Ratiolist_CM)
np.save('Ratiolist_RM_eps0.npy',Ratiolist_CM)

eps = 0.1

for k in range(1,d):

    print(k)

    NumberofPassing_CM = 0
    NumberofPassing_RM = 0

    for n in range(Samples):

        rho_temp = RandomState(eps,d)
        NumberofPassing_CM = NumberofPassing_CM + correlationmatrx_criterion(rho_temp,k,Paulis)
        NumberofPassing_RM = NumberofPassing_RM + k_reduction_criterion(rho_temp,k)

    Ratiolist_CM[k-1] = NumberofPassing_CM/Samples 
    Ratiolist_RM[k-1] = NumberofPassing_RM/Samples 


np.save('Ratiolist_CM_eps01.npy',Ratiolist_CM)
np.save('Ratiolist_RM_eps01.npy',Ratiolist_RM)

eps = 0.5

for k in range(1,d):

    print(k)

    NumberofPassing_CM = 0
    NumberofPassing_RM = 0

    for n in range(Samples):

        rho_temp = RandomState(eps,d)
        NumberofPassing_CM = NumberofPassing_CM + correlationmatrx_criterion(rho_temp,k,Paulis)
        NumberofPassing_RM = NumberofPassing_RM + k_reduction_criterion(rho_temp,k)

    Ratiolist_CM[k-1] = NumberofPassing_CM/Samples 
    Ratiolist_RM[k-1] = NumberofPassing_RM/Samples 


np.save('Ratiolist_CM_eps05.npy',Ratiolist_CM)
np.save('Ratiolist_RM_eps05.npy',Ratiolist_RM)


