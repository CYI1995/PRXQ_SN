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
    'legend.fontsize': 14,         # 强制图例保持 14
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
d_list = np.linspace(7,22,16)
CM_list = np.load('CM.npy')
RM3_list = np.load('RM3.npy')
RM4_list = np.load('RM4.npy')
RM5_list = np.load('RM5.npy')
RM6_list = np.load('RM6.npy')
RM7_list = np.load('RM7.npy')
RM8_list = np.load('RM8.npy')

plt.figure(figsize=(8.0, 4.8))
plt.xlabel(r'$\boldsymbol{d}$')
plt.ylabel(r'$\boldsymbol{\mathrm{Ratio}}$')
plt.plot(d_list, RM3_list,   label=r'$\boldsymbol{N = 3}$',   c='royalblue',      marker='o')
plt.plot(d_list, RM4_list,   label=r'$\boldsymbol{N = 4}$',   c='orange',      marker='s')
plt.plot(d_list, RM5_list,   label=r'$\boldsymbol{N = 5}$',   c='red',      marker='d')
plt.plot(d_list, RM6_list,   label=r'$\boldsymbol{N = 6}$',   c='green',      marker='^')
plt.plot(d_list, RM7_list,   label=r'$\boldsymbol{N = 7}$',   c='purple',      marker='v')
plt.plot(d_list, RM8_list,   label=r'$\boldsymbol{N = 8}$',   markerfacecolor='none', markeredgecolor='deepskyblue',   c='deepskyblue',   marker='o')
plt.plot(d_list, CM_list,   label=r'$\boldsymbol{\mathrm{CM}}$',  markerfacecolor='none', markeredgecolor='blue',  c='blue',      marker='s')

plt.legend(bbox_to_anchor=(1.05,0.55), loc='upper left', borderaxespad=0)

ax = plt.gca()
ax.set_xticks(np.arange(8, 22.1, 2))
ax.set_xticklabels([rf'$\boldsymbol{{{int(x)}}}$' for x in np.arange(8, 22.1, 2)])
ax.set_yticks(np.arange(0, 0.201, 0.05))       # 1.01 防止浮点误差漏掉 1.0

# 关键一行：让每个刻度都变成 \boldsymbol{...} 粗斜体
ax.set_yticklabels([rf'$\boldsymbol{{{val:.2f}}}$'
                    for val in ax.get_yticks()])
plt.tight_layout()
plt.savefig('FIG7.pdf', bbox_inches='tight', pad_inches=0.02, dpi=600)
plt.show()