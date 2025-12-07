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





r = 4
d = 4 
vec = np.zeros(d*d)
for i in range(r):
    vec[(d+1)*i] = 1/math.sqrt(r)
I_B = np.identity(d)
rho = np.outer(vec,vec.conj())


N = 20
eps_list = np.linspace(0,1,N)
knegetivity_list = np.zeros(N)

k = 1
for n in range(N):

    eps = eps_list[n]

    rho_temp = (1-eps)*rho + eps*np.identity(d*d)/(d*d)
    rho_A = partial_trace(rho_temp, d, d)
    Rkrho = k*np.kron(rho_A,I_B) - rho_temp
    eig,vec = np.linalg.eig(Rkrho)

    kn_temp = (-1)*np.min(eig).real

    if(kn_temp > 0):
        knegetivity_list[n] = kn_temp 
    else:
        knegetivity_list[n] = 0

np.save('kn_list1.npy',knegetivity_list)


knegetivity_list = np.zeros(N)

k = 2
for n in range(N):

    eps = eps_list[n]

    rho_temp = (1-eps)*rho + eps*np.identity(d*d)/(d*d)
    rho_A = partial_trace(rho_temp, d, d)
    Rkrho = k*np.kron(rho_A,I_B) - rho_temp
    eig,vec = np.linalg.eig(Rkrho)

    kn_temp = (-1)*np.min(eig).real

    if(kn_temp > 0):
        knegetivity_list[n] = kn_temp 
    else:
        knegetivity_list[n] = 0

np.save('kn_list2.npy',knegetivity_list)

knegetivity_list = np.zeros(N)

k = 3
for n in range(N):

    eps = eps_list[n]

    rho_temp = (1-eps)*rho + eps*np.identity(d*d)/(d*d)
    rho_A = partial_trace(rho_temp, d, d)
    Rkrho = k*np.kron(rho_A,I_B) - rho_temp
    eig,vec = np.linalg.eig(Rkrho)

    kn_temp = (-1)*np.min(eig).real

    if(kn_temp > 0):
        knegetivity_list[n] = kn_temp 
    else:
        knegetivity_list[n] = 0

np.save('kn_list3.npy',knegetivity_list)


