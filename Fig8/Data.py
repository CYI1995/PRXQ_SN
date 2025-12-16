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

def RandomState(eps,d):

    M = d*d  
    z = np.random.normal(0, 1, M) + 1j * np.random.normal(0, 1, M)
    v = z / np.linalg.norm(z)

    return (1-eps) * np.outer(v, v.conj()) + eps * np.identity(d*d)/(d*d) 


SampleNum = 100

order_list = np.array([3,5,7,9])
k_list = np.array([1,2,3,4,5,6,7])


L1 = len(k_list)
L2 = len(order_list)
d = 8
ListofRatio = np.zeros((L1,L2))

for n in range(SampleNum):
    print(n)
    rho = RandomState(0,d)
    for l in range(L2):
        order = order_list[l]
        for j in range(L1):
            k = k_list[j]
            temp = moment_based_k_reduction_criterion(rho,k,d,order)
            ListofRatio[j][l] = ListofRatio[j][l] + temp

ListofRatio = ListofRatio/SampleNum
np.save('RMd8.npy',ListofRatio)


d = 16
ListofRatio = np.zeros((L1,L2))

for n in range(SampleNum):
    print(n)
    rho = RandomState(0,d)
    for l in range(L2):
        order = order_list[l]
        for j in range(L1):
            k = k_list[j]
            temp = moment_based_k_reduction_criterion(rho,k,d,order)
            ListofRatio[j][l] = ListofRatio[j][l] + temp

ListofRatio = ListofRatio/SampleNum
np.save('RMd16.npy',ListofRatio)

