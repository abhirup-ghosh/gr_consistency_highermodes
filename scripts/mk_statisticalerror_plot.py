import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import plotsettings
from matplotlib import rc
mpl.rc('text.latex', preamble = '\usepackage{txfonts}')
import bayesian as ba
import imrtestgr as tgr
import scipy
from scipy import interpolate

# # Make the font match the document
# rc_params = {'backend': 'ps',
#              'axes.labelsize': 15,
#              'axes.titlesize': 15,
#              'font.size': 15,
#              'legend.fontsize': 15,
#              'xtick.labelsize': 15,
#              'ytick.labelsize': 15,
#              'font.family': 'Times New Roman',
#              'font.family': 'sans-serif',
#              'font.sans-serif': ['Bitstream Vera Sans']
#              }

col = 'k'

fname_vec_q = ['../data/90_percent_width_dMc_t_abhi.txt', '../data/90_percent_width_dq_t_abhi.txt']
fname_vec_M = ['../data/90_percent_width_dMc_t_diffM_abhi.txt', '../data/90_percent_width_dq_t_diffM_abhi.txt']
fname_vec_q = ['../data/90_percent_width_dMc_t_sidharth.txt', '../data/90_percent_width_dq_t_sidharth.txt']

plt.figure(figsize=(4,4))

for i, fname in enumerate(fname_vec_q):

	q, i_30, i_45, i_60, i_80, i_90 = np.loadtxt(fname, unpack=True)

	ax = plt.subplot(2,1,i+1)
	ax.semilogy(q, i_30, color='lightgray', marker='^', markeredgecolor='lightgray',label='$\iota = 30^{\circ}$', alpha=1)
        ax.semilogy(q, i_45, color='lightpink', marker='s',  markeredgecolor='lightpink', label='$\iota = 45^{\circ}$', alpha=1)
        ax.semilogy(q, i_60, color='crimson', marker='o',  markeredgecolor='crimson', label='$\iota = 60^{\circ}$', alpha=1)
        ax.semilogy(q, i_80, color='k', marker='d',  markeredgecolor='k', label='$\iota = 80^{\circ}$', alpha=1, lw=1, ls='--')
        ax.semilogy(q, i_90, color='deeppink', marker='v',  markeredgecolor='deeppink', label='$\iota = 90^{\circ}$', alpha=1, lw=1)
	ax.set_ylim(1e-3,1e-1)
	ax.set_xlim(2, 9)

	if i == 0:
		ax.set_ylabel('90\% width of $\Delta M_c/M_c$', labelpad=5, fontsize=10)
		plt.legend(loc='best', frameon=False, fontsize=8)
	else: 
		ax.set_ylabel('90\% width of $\Delta q$', labelpad=7, fontsize=10)

ax.set_xlabel('Mass ratio $q$', labelpad=5, fontsize=10)
plt.tight_layout()
plt.savefig('../papers/intro_paper/figs/fig3a.png', dpi=300)
plt.savefig('../papers/intro_paper/figs/fig3a.pdf')
plt.close()

plt.figure(figsize=(4,4))

for i, fname in enumerate(fname_vec_M):

        M, i_30, i_45, i_60, i_80, i_90 = np.loadtxt(fname, unpack=True)

        ax = plt.subplot(2,1,i+1)
        ax.semilogy(M, i_30, color='lightgray', marker='^', markeredgecolor='lightgray',label='$\iota = 30^{\circ}$', alpha=1)
        ax.semilogy(M, i_45, color='lightpink', marker='s',  markeredgecolor='lightpink', label='$\iota = 45^{\circ}$', alpha=1)
        ax.semilogy(M, i_60, color='crimson', marker='o',  markeredgecolor='crimson', label='$\iota = 60^{\circ}$', alpha=1)
        ax.semilogy(M, i_80, color='k', marker='d',  markeredgecolor='k', label='$\iota = 80^{\circ}$', alpha=1, lw=1, ls='--')
        ax.semilogy(M, i_90, color='deeppink', marker='v',  markeredgecolor='deeppink', label='$\iota = 90^{\circ}$', alpha=1, lw=1)

        if i == 0:
        	ax.set_ylim(1e-4,1e-2)
                ax.set_ylabel('90\% width of $\Delta M_c/M_c$', labelpad=5, fontsize=10)
                plt.legend(loc='best', frameon=False, fontsize=8)
        else:
                ax.set_ylim(1e-3, 1e-2)
		ax.set_ylabel('90\% width of $\Delta q$', labelpad=7, fontsize=10)

ax.set_xlabel('Total mass $M (M_{\odot})$', labelpad=5, fontsize=10)
plt.tight_layout()
plt.savefig('../papers/intro_paper/figs/fig3b.png', dpi=300)
plt.savefig('../papers/intro_paper/figs/fig3b.pdf')
plt.close()
