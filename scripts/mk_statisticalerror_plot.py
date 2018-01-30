import matplotlib as mpl
#mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
#import plotsettings
from optparse import OptionParser
from matplotlib import rc
mpl.rc('text.latex', preamble = '\usepackage{txfonts}')
import bayesian as ba
import imrtestgr as tgr
import scipy
from scipy import interpolate

# Make the font match the document
rc_params = {'backend': 'ps',
             'axes.labelsize': 10,
             'axes.titlesize': 10,
             'font.size': 12,
             'legend.fontsize': 12,
             'xtick.labelsize': 13,
             'ytick.labelsize': 13,
             'font.family': 'Times New Roman',
             'font.family': 'sans-serif',
             'font.sans-serif': ['Bitstream Vera Sans']
             }

color = ['#0072b2', '#CC79A7']

post_loc = '../data/90_percent_width_dMc_t.txt'
q, i_30, i_45, i_60, i_80, i_90 = np.loadtxt(post_loc, unpack=True)

plt.figure(figsize=(5,5))
ax = plt.subplot(111)
ax.semilogy(q, i_30, color='r', marker='o', label='$\iota = 30^{\circ}$')
ax.semilogy(q, i_45, color='orange', marker='o', label='$\iota = 45^{\circ}$')
ax.semilogy(q, i_60, color='g', marker='o', label='$\iota = 60^{\circ}$')
ax.semilogy(q, i_80, color='b', marker='o', label='$\iota = 80^{\circ}$')
ax.semilogy(q, i_90, color='k', marker='o', label='$\iota = 90^{\circ}$')
plt.legend(loc='best', frameon=False, fontsize=12)
ax.set_xlabel('Asymmetric Mass Ratio (q)', labelpad=10, fontsize=12)
ax.set_ylabel('90% width of $\Delta M_c/M_c$', labelpad=10, fontsize=12)
plt.tight_layout()
plt.savefig('../papers/intro_paper/figs/fig3.png', dpi=300)
plt.savefig('../papers/intro_paper/figs/fig3.pdf')
plt.show()
