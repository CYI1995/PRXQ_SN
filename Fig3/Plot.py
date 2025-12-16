import numpy  as np
import scipy
import math
from scipy import linalg
import source as mycode
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
klist = np.array([1,2,3,4,5,6,7])
ylist = np.array([0.0,0.2,0.4,0.6,0.8,1.0])

Ratiolist_CM_eps0 = np.load('Ratiolist_CM_eps0.npy')
Ratiolist_RM_eps0 = np.load('Ratiolist_RM_eps0.npy')

Ratiolist_CM_eps01 = np.load('Ratiolist_CM_eps01.npy')
Ratiolist_RM_eps01 = np.load('Ratiolist_RM_eps01.npy')

Ratiolist_CM_eps05 = np.load('Ratiolist_CM_eps05.npy')
Ratiolist_RM_eps05 = np.load('Ratiolist_RM_eps05.npy')

plt.figure(figsize=(7.0, 5.2))
plt.xlabel(r'$\boldsymbol{k}$')
plt.ylabel(r'$\boldsymbol{\mathrm{Ratio}}$')
plt.plot(klist, Ratiolist_CM_eps0,  '--',   c='royalblue',      marker='o')
plt.plot(klist, Ratiolist_RM_eps0,   label=r'$\boldsymbol{\varepsilon = 0}$',         c='royalblue',      marker='o')
plt.plot(klist, Ratiolist_CM_eps01, '--',  c='orange', marker='s')
plt.plot(klist, Ratiolist_RM_eps01,  label=r'$\boldsymbol{\varepsilon = 0.1}$',        c='orange', marker='s')
plt.plot(klist, Ratiolist_CM_eps05, '--',  c='g',      marker='^')
plt.plot(klist, Ratiolist_RM_eps05,   label=r'$\boldsymbol{\varepsilon = 0.5}$',       c='g',      marker='^')
plt.legend()

ax = plt.gca()
ax.set_xticks(klist)                                # 确保刻度就是 1,2,3...
ax.set_xticklabels([rf'$\boldsymbol{{{int(x)}}}$' for x in klist])
ax.set_yticks(np.arange(0, 1.01, 0.2))       # 1.01 防止浮点误差漏掉 1.0
ax.set_yticklabels([rf'$\boldsymbol{{{val:.1f}}}$' 
                    for val in ax.get_yticks()])
plt.tight_layout()
plt.savefig('FIG3.pdf', bbox_inches='tight', pad_inches=0.02, dpi=600)
plt.show()