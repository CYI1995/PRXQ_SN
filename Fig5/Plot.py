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
N1list = np.linspace(3,16,14)
N2list = np.linspace(3,32,30)

RMd8 = np.load('RMd8.npy')
RMd16 = np.load('RMd16.npy')

RM_list1 = RMd8[0,:]
RM_list2 = RMd8[1,:]
RM_list3 = RMd8[2,:]
RM_list4 = RMd8[3,:]
RM_list5 = RMd8[4,:]

RM_list6 = RMd16[0,:]
RM_list7 = RMd16[1,:]
RM_list8 = RMd16[2,:]
RM_list9 = RMd16[3,:]
RM_list10 = RMd16[4,:]

# ============================== 上下两张子图 ==============================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 9.8),
                               gridspec_kw={'hspace': 0.35})   # 两图间距超小

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
markers = ['o', 's', 'd', '^', 'v']

for i, data in enumerate([RM_list1, RM_list2, RM_list3, RM_list4, RM_list5], 1):
    ax1.plot(N1list, data, marker=markers[i-1], color=colors[i-1],
             lw=1.8, ms=7, label=rf'$\boldsymbol{{k = {i}}}$')

ax1.set_xticks(np.arange(3, 16, 2))
ax1.set_xticklabels([rf'$\boldsymbol{{{int(x)}}}$' for x in np.arange(3, 16, 2)])
ax1.set_ylabel(r'$\boldsymbol{\mathrm{Ratio}}$')

ax1.legend(loc='lower right')

for i, data in enumerate([RM_list6, RM_list7, RM_list8, RM_list9, RM_list10], 1):
    ax2.plot(N2list, data, marker=markers[i-1], color=colors[i-1],
             lw=1.8, ms=7, label=rf'$\boldsymbol{{k = {i}}}$')

ax2.set_xticks(np.arange(3, 32, 4))   # 保留这行！
ax2.set_xticklabels([rf'$\boldsymbol{{{int(x)}}}$' for x in np.arange(3, 32, 4)])

ax2.set_xlabel(r'$\boldsymbol{N}$')
ax2.set_ylabel(r'$\boldsymbol{\mathrm{Ratio}}$')
ax2.legend(loc='lower right')


for ax in (ax1, ax2):
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_yticklabels([rf'$\boldsymbol{{{y:.1f}}}$' 
                        for y in np.arange(0, 1.01, 0.2)])
    ax.set_ylim(-0.02, 1.02)   # 可选：让两图 y 范围完全一致

plt.savefig('FIG5.pdf', bbox_inches='tight', pad_inches=0.02, dpi=600)
plt.show()