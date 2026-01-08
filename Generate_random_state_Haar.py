import numpy  as np
import scipy
import math
from scipy import linalg
import matplotlib
from matplotlib import pyplot as plt


d = 8

Samples = 10000
RandomVectors = []

M = d*d 

for n in range(Samples):
    print(n)

    z = np.random.normal(0, 1, M) + 1j * np.random.normal(0, 1, M)
    v = z / np.linalg.norm(z)
    RandomVectors.append(v)

np.save('RandomVectors_d8.npy',RandomVectors)

d = 16

Samples = 10000
RandomVectors = []

M = d*d 

for n in range(Samples):
    print(n)

    z = np.random.normal(0, 1, M) + 1j * np.random.normal(0, 1, M)
    v = z / np.linalg.norm(z)
    RandomVectors.append(v)

np.save('RandomVectors_d16.npy',RandomVectors)








