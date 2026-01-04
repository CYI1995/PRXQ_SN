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
    





d = 16
Samples = 20

K = 2
RandomStates = []
M = d*d*K
for n in range(Samples):
    print(n)
    z = np.random.normal(0, 1, M) + 1j * np.random.normal(0, 1, M)
    v = z / np.linalg.norm(z)
    psi = np.outer(v,v.conj())
    rho = partial_trace(psi,d*d,K)
    RandomStates.append(rho)

np.save('RMSK2.npy',RandomStates)

K = 3
RandomStates = []
M = d*d*K
for n in range(Samples):
    print(n)

    z = np.random.normal(0, 1, M) + 1j * np.random.normal(0, 1, M)
    v = z / np.linalg.norm(z)
    psi = np.outer(v,v.conj())
    rho = partial_trace(psi,d*d,K)
    RandomStates.append(rho)

np.save('RMSK3.npy',RandomStates)

K = 4
RandomStates = []
M = d*d*K
for n in range(Samples):
    print(n)

    z = np.random.normal(0, 1, M) + 1j * np.random.normal(0, 1, M)
    v = z / np.linalg.norm(z)
    psi = np.outer(v,v.conj())
    rho = partial_trace(psi,d*d,K)
    RandomStates.append(rho)

np.save('RMSK4.npy',RandomStates)

K = 5
RandomStates = []
M = d*d*K
for n in range(Samples):
    print(n)

    z = np.random.normal(0, 1, M) + 1j * np.random.normal(0, 1, M)
    v = z / np.linalg.norm(z)
    psi = np.outer(v,v.conj())
    rho = partial_trace(psi,d*d,K)
    RandomStates.append(rho)

np.save('RMSK5.npy',RandomStates)

K = 6
RandomStates = []
M = d*d*K
for n in range(Samples):
    print(n)

    z = np.random.normal(0, 1, M) + 1j * np.random.normal(0, 1, M)
    v = z / np.linalg.norm(z)
    psi = np.outer(v,v.conj())
    rho = partial_trace(psi,d*d,K)
    RandomStates.append(rho)

np.save('RMSK6.npy',RandomStates)


