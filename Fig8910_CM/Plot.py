import numpy  as np
import scipy
import math
from scipy import linalg
import matplotlib
from matplotlib import pyplot as plt
import source as mycode 


L1 = np.load('Ratiolist_CM_d8e0.npy')
L2 = np.load('Ratiolist_CM_d8e01.npy')
L3 = np.load('Ratiolist_CM_d8e05.npy')
X = np.linspace(1,7,7)

plt.plot(X,L1)
plt.plot(X,L2)
plt.plot(X,L3)
plt.show()
