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

def k_reduction_criterion(rho,k,d):

    #Rk(rho) \ge 0, return 0; else return 1.

    I_B = np.identity(d)

    if(len(rho[0]) == d*d):

        rho_A = partial_trace(rho, d, d)
        Rkrho = k*np.kron(rho_A,I_B) - rho 
        eig,vec = np.linalg.eig(Rkrho)
        if(np.min(eig) < 0):

            return 1
        else:
            return 0 

    else:

        print('Error')

        return -1


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

def moment_based_CM(T,k,d):

    #If the criterion is violated, return 1; else return 0. 

    U, S, Vh = np.linalg.svd(T, full_matrices=True)

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
    
    if(n>=d*d-1):

        fn = (d*d+1)*S2*S2/(3*(d*d-1))

        if(S4 < fn):
            return 1
        else:
            return 0

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


d = 8

Samples = 1000
Ratiolist_CM0 = np.zeros(7)
Ratiolist_CM1 = np.zeros(7)
Ratiolist_CM2 = np.zeros(7)

klist = np.array([1,2,3,4,5,6,7])


basis = np.load('Pauli_basis_d8.npy')

L = len(basis)

for n in range(Samples):
    print(n)

    rho_temp = RandomState(0,d)
    T_cor = np.zeros((L,L),dtype = complex)
    for i in range(1,L):
        Pi = basis[i]
        for j in range(1,L):
            Pj = basis[j]
            T_cor[i][j] = np.trace(rho_temp.dot(np.kron(Pi,Pj))).real/d

    for i in range(7):
        k = klist[i]

        if(moment_based_CM(T_cor,k,d) == 0):
            break 
        else:
            Ratiolist_CM0[i] = Ratiolist_CM0[i] + 1

    for i in range(7):
        k = klist[i]

        if(moment_based_CM(0.9*T_cor,k,d) == 0):
            break 
        else:
            Ratiolist_CM1[i] = Ratiolist_CM1[i] + 1

    for i in range(7):
        k = klist[i]

        if(moment_based_CM(0.5*T_cor,k,d) == 0):
            break 
        else:
            Ratiolist_CM2[i] = Ratiolist_CM2[i] + 1


Ratiolist_CM0 = Ratiolist_CM0/Samples
np.save('FIG8_CM_d8e0.npy',Ratiolist_CM0)

Ratiolist_CM1 = Ratiolist_CM1/Samples
np.save('FIG9_CM_d8e01.npy',Ratiolist_CM1)

Ratiolist_CM2 = Ratiolist_CM2/Samples
np.save('FIG10_CM_d8e05.npy',Ratiolist_CM2)








