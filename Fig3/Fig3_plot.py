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


# ██████ 关键配置开始 ██████
plt.rcParams.update({
    "text.usetex": True,                                      # 开启真 LaTeX 渲染
    "text.latex.preamble": r"""
        \usepackage{type1cm}           # 让 LaTeX 能随意缩放字体
        \usepackage{amsmath,amssymb}   # 数学符号
        \usepackage{times}             # 正文用 Times（和新罗马几乎一样）
        \usepackage{helvet}            # 无衬线用 Helvetica（图例更清晰）
        \usepackage{courier}           # 等宽字体
    """,
    "font.family": "serif",            # 重要！必须是 serif 才能被 LaTeX 接管
    "font.serif": ["Times"],           # 真正让 LaTeX 用 Times
    "font.size": 14,

    "axes.labelsize": 18,              # x/y label 放大
    "axes.titlesize": 18,
    "legend.fontsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,

    "axes.unicode_minus": False,       # 负号显示正确
    "pgf.texsystem": "xelatex",        # 可选：如果你以后要保存 pgf 的话
})
# ██████ 关键配置结束 ██████
# 下面你的原代码完全不变

klist = np.array([1,2,3,4,5,6,7])


Ratiolist_CM_eps0 = np.load('Ratiolist_CM_eps0.npy')
Ratiolist_RM_eps0 = np.load('Ratiolist_RM_eps0.npy')

Ratiolist_CM_eps01 = np.load('Ratiolist_CM_eps01.npy')
Ratiolist_RM_eps01 = np.load('Ratiolist_RM_eps01.npy')

Ratiolist_CM_eps05 = np.load('Ratiolist_CM_eps05.npy')
Ratiolist_RM_eps05 = np.load('Ratiolist_RM_eps05.npy')

plt.figure(figsize=(8, 5.5))                    # 论文常用比例
plt.xlabel(r'$k$')
plt.ylabel(r'$\mathrm{Ratio}$')                # 加 \mathrm 让 Ratio 罗马体（更标准）

plt.plot(klist, Ratiolist_CM_eps0,  '--', label=r'$\varepsilon = 0$',   c='royalblue',      marker='o')
plt.plot(klist, Ratiolist_RM_eps0,           c='royalblue',      marker='o')
plt.plot(klist, Ratiolist_CM_eps01, '--', label=r'$\varepsilon = 0.1$', c='orange', marker='s')
plt.plot(klist, Ratiolist_RM_eps01,          c='orange', marker='s')
plt.plot(klist, Ratiolist_CM_eps05, '--', label=r'$\varepsilon = 0.5$', c='g',      marker='^')
plt.plot(klist, Ratiolist_RM_eps05,          c='g',      marker='^')
plt.legend()
plt.show()