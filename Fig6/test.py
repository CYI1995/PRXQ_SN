import numpy  as np
import scipy
import math
from scipy import linalg

import matplotlib.pyplot as plt


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.unicode_minus'] = False

# 关键三行（必须同时设置！）
plt.rcParams['text.usetex'] = True                          # 开启真 LaTeX
plt.rcParams['text.latex.preamble'] = r'''

'''
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 18,               # 全局默认
    'axes.labelsize': 18,          # 只放大坐标轴标签 ← 你现在最想要的
    'legend.fontsize': 18,         # 强制图例保持 14
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    "font.weight": "bold",           # 普通文本默认粗体
    "axes.labelweight": "bold",      # xlabel/ylabel 粗体
    "axes.titleweight": "bold",      # title 粗体
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{times}\usepackage{amsmath,amssymb}',
    'axes.unicode_minus': False,
})

# 下面你的原代码完全不变
N1list = np.linspace(3,16,14)
N2list = np.linspace(3,31,29)
RM_list1 = np.load('RMd8k1.npy')
RM_list2 = np.load('RMd8k2.npy')
RM_list3 = np.load('RMd8k3.npy')
RM_list4 = np.load('RMd8k4.npy')
RM_list5 = np.load('RMd8k5.npy')
RM_list6 = np.load('RMd16k1.npy')
RM_list7 = np.load('RMd16k2.npy')
RM_list8 = np.load('RMd16k3.npy')
RM_list9 = np.load('RMd16k4.npy')
RM_list10 = np.load('RMd16k5.npy')


plt.plot(N2list,RM_list10[:29])
plt.show()