import numpy  as np
import scipy
import math
from scipy import linalg
import matplotlib
import random
from matplotlib import pyplot as plt
import source as mycode 

def symplectic_inner_product(string1,string2,N):

    output = 0

    for i in range(N):
        output = output + string1[i]*string2[i+N]
        output = output + string2[i]*string1[i+N] 

    return output%2 

def Action_Weyl(x,z,string,N):

    new_string = np.zeros(N)
    sgn = 0

    for i in range(N):
        x_bit = x[i]
        z_bit = z[i]
        new_string[i] = (string[i] + x_bit)%2 
        sgn = sgn + string[i]*z_bit 

    return (-1)**(sgn%2), new_string

def Weyl(x,z,N):

    dim = 2**N

    phase_angle = (int(np.vdot(x,z)%4))*math.pi/2
    phase = math.cos(phase_angle) + 1j*math.sin(phase_angle)

    P = np.zeros((dim,dim),dtype = complex)
    for i in range(dim):
        bit_i = mycode.dec_to_bin(i,N)
        sgn,new_bit = Action_Weyl(x,z,bit_i,N)
        new_i = mycode.bin_to_dec(new_bit,N)
        P[new_i][i] = sgn*phase
        # print(i,new_i)

    return P 


N = 3
dim = 2**N

Pauli_basis = []
string_Pauli_basis = []

for x in range(dim):
    x_string = mycode.dec_to_bin(x,N)
    for z in range(dim):
        z_string = mycode.dec_to_bin(z,N)
        xz_string = mycode.dec_to_bin(x*dim+z,2*N)
        P = Weyl(x_string,z_string,N)
        Pauli_basis.append(P)
        string_Pauli_basis.append(xz_string)

np.save('Pauli_basis.npy',Pauli_basis)


# sampling_ratio = 32/(dim*dim)
# S = []
# for i in range(dim*dim):

#     r = random.uniform(0,1)
#     if(r < sampling_ratio):
#         S.append(string_Pauli_basis[i])

# np.save('random_Pauli_sampling.npy',S)