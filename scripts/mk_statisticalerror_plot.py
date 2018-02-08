import matplotlib as mpl
#mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import plotsettings
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


fname_vec = ['../data/90_percent_width_dMc_t_abhi.txt', '../data/90_percent_width_dq_t_abhi.txt']

plt.figure(figsize=(5,5))

for i, fname in enumerate(fname_vec):

	q, i_30, i_45, i_60, i_80, i_90 = np.loadtxt(fname, unpack=True)

	ax = plt.subplot(2,1,i+1)
	ax.semilogy(q, i_30, color='r', marker='^', label='$\iota = 30^{\circ}$')
	ax.semilogy(q, i_45, color='orange', marker='s', label='$\iota = 45^{\circ}$')
	ax.semilogy(q, i_60, color='lawngreen', marker='o', label='$\iota = 60^{\circ}$')
	ax.semilogy(q, i_80, color='cyan', marker='d', label='$\iota = 80^{\circ}$')
	ax.semilogy(q, i_90, color='blue', marker='v', label='$\iota = 90^{\circ}$')
	plt.legend(loc='best', frameon=False, fontsize=12)

	if i == 0:
		ax.set_ylabel('90\% width of $\Delta M_c/M_c$', labelpad=10, fontsize=12)
	else: 
		ax.set_ylabel('90\% width of $\Delta q$', labelpad=10, fontsize=12)

ax.set_xlabel('Mass ratio $q$', labelpad=10, fontsize=12)
plt.tight_layout()
plt.savefig('../papers/intro_paper/figs/fig3.png', dpi=300)
plt.savefig('../papers/intro_paper/figs/fig3.pdf')
plt.close()
