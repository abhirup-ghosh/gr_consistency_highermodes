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

f, hf_gr, hf_modgr = np.loadtxt('../data/data_amplitude_vs_freq.txt', unpack=True)

plt.figure(figsize=(5,5))
ax1 = plt.subplot(111)
ax1.loglog(f, hf_gr, color=color[0], label='$GR$')
ax1.loglog(f, hf_modgr, color=color[1], lw=2, alpha=0.5, label='$modGR$')
ax1.set_xlabel('$f$ $(Hz)$', fontsize=14)
ax1.set_ylabel('$|\\tilde{h}(f)|$', fontsize=14)
ax1.set_xlim([20, 1e3])
ax1.set_ylim([1e-25, 1e-22])
ax1.tick_params(axis='both', which='major', labelsize=12)
plt.legend(loc='best', frameon=False, fontsize=14)
plt.tight_layout()
plt.savefig('../papers/intro_paper/figs/fig2.png', dpi=300)
plt.savefig('../papers/intro_paper/figs/fig2.pdf')
plt.show()
