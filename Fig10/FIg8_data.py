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

def moment_based_correlationmatrx_criterion(rho,k,Paulis,d):

    #If the criterion is violated, return 1; else return 0. 

    L = d*d

    T_cor = np.zeros((L,L),dtype = complex)
    for i in range(1,L):
        P1 = Paulis[i]
        for j in range(1,L):
            P2 = Paulis[j]
            T_cor[i][j] = mycode.trace(rho.dot(np.kron(P1,P2)))/d

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


d = 16

Samples = 1000
Ratiolist_RM3 = np.zeros(7)
Ratiolist_RM5 = np.zeros(7)
Ratiolist_RM7 = np.zeros(7)
Ratiolist_RM9 = np.zeros(7)
Ratiolist_CM = np.zeros(7)
klist = np.array([1,2,3,4,5,6,7])

eps = 0.5

for n in range(Samples):

    print(n)

    rho_temp = RandomState(eps,d)

    for i in range(7):
        k = klist[i]
        if(moment_based_k_reduction_criterion(rho_temp,k,d,3) == 0):
            break 
        else:
            Ratiolist_RM3[i] = Ratiolist_RM3[i] + 1

    for i in range(7):
        k = klist[i]
        if(moment_based_k_reduction_criterion(rho_temp,k,d,5) == 0):
            break 
        else:
            Ratiolist_RM5[i] = Ratiolist_RM5[i] + 1

    for i in range(7):
        k = klist[i]
        if(moment_based_k_reduction_criterion(rho_temp,k,d,7) == 0):
            break 
        else:
            Ratiolist_RM7[i] = Ratiolist_RM7[i] + 1

    for i in range(7):
        k = klist[i]
        if(moment_based_k_reduction_criterion(rho_temp,k,d,9) == 0):
            break 
        else:
            Ratiolist_RM9[i] = Ratiolist_RM9[i] + 1

Ratiolist_RM3 = Ratiolist_RM3/Samples
Ratiolist_RM5 = Ratiolist_RM5/Samples
Ratiolist_RM7 = Ratiolist_RM7/Samples
Ratiolist_RM9 = Ratiolist_RM9/Samples


np.save('Ratiolist_RM3_d16e05.npy',Ratiolist_RM3)
np.save('Ratiolist_RM5_d16e05.npy',Ratiolist_RM5)
np.save('Ratiolist_RM7_d16e05.npy',Ratiolist_RM7)
np.save('Ratiolist_RM9_d16e05.npy',Ratiolist_RM9)

# np.save('Ratiolist_RM3_d16e0.npy',Ratiolist_RM3)
# np.save('Ratiolist_RM5_d16e0.npy',Ratiolist_RM5)
# np.save('Ratiolist_RM7_d16e0.npy',Ratiolist_RM7)
# np.save('Ratiolist_RM9_d16e0.npy',Ratiolist_RM9)

plt.plot(klist,Ratiolist_RM3,label = 'N=3')
plt.plot(klist,Ratiolist_RM5,label = 'N=5')
plt.plot(klist,Ratiolist_RM7,label = 'N=7')
plt.plot(klist,Ratiolist_RM9,label = 'N=9')
plt.legend()
plt.show()







