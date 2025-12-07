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
eps_list = np.linspace(0,1,20)
kn_list1 = np.load('kn_list1.npy')
kn_list2 = np.load('kn_list2.npy')
kn_list3 = np.load('kn_list3.npy')

plt.figure(figsize=(7.0, 4.8))
plt.xlabel(r'$\boldsymbol{\varepsilon}$')
plt.ylabel(r'$\boldsymbol{\mathcal{N}_k(\rho_{\varepsilon,r})}$')
plt.plot(eps_list, kn_list1,   label=r'$\boldsymbol{k = 1}$',   c='royalblue',      marker='o')
plt.plot(eps_list, kn_list2,  label=r'$\boldsymbol{k = 2}$', c='orange', marker='s')
plt.plot(eps_list, kn_list3, label=r'$\boldsymbol{k = 3}$', c='red',      marker='d')

plt.legend()

ax = plt.gca()
ax.set_xticks(np.arange(0, 1.01, 0.2))       # 1.01 防止浮点误差漏掉 1.0

# 关键一行：让每个刻度都变成 \boldsymbol{...} 粗斜体
ax.set_xticklabels([rf'$\boldsymbol{{{val:.1f}}}$' 
                    for val in ax.get_xticks()])
ax.set_yticks(np.arange(0, 0.81, 0.2))       # 1.01 防止浮点误差漏掉 1.0

# 关键一行：让每个刻度都变成 \boldsymbol{...} 粗斜体
ax.set_yticklabels([rf'$\boldsymbol{{{val:.1f}}}$'
                    for val in ax.get_yticks()])
plt.tight_layout()
plt.savefig('FIG1.pdf', bbox_inches='tight', pad_inches=0.02, dpi=600)
plt.show()