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
klist = np.linspace(1,7,7)
print(klist)
RM_list1 = np.load('Ratiolist_RM3_d8e0.npy')
RM_list2 = np.load('Ratiolist_RM5_d8e0.npy')
RM_list3 = np.load('Ratiolist_RM7_d8e0.npy')
RM_list4 = np.load('Ratiolist_RM9_d8e0.npy')
RM_list5 = np.load('Ratiolist_RM3_d16e0.npy')
RM_list6 = np.load('Ratiolist_RM5_d16e0.npy')
RM_list7 = np.load('Ratiolist_RM7_d16e0.npy')
RM_list8 = np.load('Ratiolist_RM9_d16e0.npy')
CM_list1 = np.load('Ratiolist_CM_d8e0.npy')
CM_list2 = np.load('Ratiolist_CM_d16e0.npy')


# ============================== 上下两张子图 ==============================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 9.8),
                               gridspec_kw={'hspace': 0.35})   # 两图间距超小

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
markers = ['o', 's', 'd', '^', 'v']

for i, data in enumerate([RM_list1, RM_list2, RM_list3, RM_list4], 1):
    ax1.plot(klist, data, marker=markers[i-1], color=colors[i-1],
             lw=1.8, ms=7, label=rf'$\boldsymbol{{N = {2*i+1}}}$')
    
ax1.plot(klist, CM_list1, marker=markers[4], color=colors[4],
             lw=1.8, ms=7, label=r'$\boldsymbol{\mathrm{CM}}$')

ax1.set_xticks(np.arange(1, 7.1, 1))
ax1.set_xticklabels([rf'$\boldsymbol{{{int(x)}}}$' for x in np.arange(1, 7.1, 1)])
ax1.set_ylabel(r'$\boldsymbol{\mathrm{Ratio}}$')

ax1.legend(loc='upper right')

for i, data in enumerate([RM_list5,RM_list6, RM_list7, RM_list8], 1):
    ax2.plot(klist, data, marker=markers[i-1], color=colors[i-1],
             lw=1.8, ms=7, label=rf'$\boldsymbol{{N = {2*i+1}}}$')

ax2.plot(klist, CM_list2, marker=markers[4], color=colors[4],
             lw=1.8, ms=7, label=r'$\boldsymbol{\mathrm{CM}}$')

ax2.set_xticks(np.arange(1, 7.1, 1))   # 保留这行！
ax2.set_xticklabels([rf'$\boldsymbol{{{int(x)}}}$' for x in np.arange(1, 7.1, 1)])

ax2.set_xlabel(r'$\boldsymbol{k}$')
ax2.set_ylabel(r'$\boldsymbol{\mathrm{Ratio}}$')
ax2.legend(loc='lower left')


for ax in (ax1, ax2):
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_yticklabels([rf'$\boldsymbol{{{y:.1f}}}$' 
                        for y in np.arange(0, 1.01, 0.2)])
    ax.set_ylim(-0.02, 1.02)   # 可选：让两图 y 范围完全一致

plt.savefig('FIG8.pdf', bbox_inches='tight', pad_inches=0.02, dpi=600)
plt.show()