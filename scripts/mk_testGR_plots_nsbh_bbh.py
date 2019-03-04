import sys
sys.path.append('../src')
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import plotsettings
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

# Module for confidence calculations
class confidence(object):
  def __init__(self, counts):
    # Sort in descending order in frequency
    self.counts_sorted = np.sort(counts.flatten())[::-1]
    # Get a normalized cumulative distribution from the mode
    self.norm_cumsum_counts_sorted = np.cumsum(self.counts_sorted) / np.sum(counts)
    # Set interpolations between heights, bins and levels
    self._set_interp()
  def _set_interp(self):
    self._length = len(self.counts_sorted)
    # height from index
    self._height_from_idx = interpolate.interp1d(np.arange(self._length), self.counts_sorted, bounds_error=False, fill_value=0.)
    # index from height
    self._idx_from_height = interpolate.interp1d(self.counts_sorted[::-1], np.arange(self._length)[::-1], bounds_error=False, fill_value=self._length)
    # level from index
    self._level_from_idx = interpolate.interp1d(np.arange(self._length), self.norm_cumsum_counts_sorted, bounds_error=False, fill_value=1.)
    # index from level
    self._idx_from_level = interpolate.interp1d(self.norm_cumsum_counts_sorted, np.arange(self._length), bounds_error=False, fill_value=self._length)
  def level_from_height(self, height):
    return self._level_from_idx(self._idx_from_height(height))
  def height_from_level(self, level):
    return self._height_from_idx(self._idx_from_level(level))

# compute 1-sigma confidence intervals in 1D 
def calc_cred_intervals_in_1d(P, x):

        # find the value of P corresponding to 50% and 9% confidence heights 
        conf = confidence(P)
        P_s1 = conf.height_from_level(0.5)
        P_s2 = conf.height_from_level(0.9)

        # calculation of condifence edges (values of x corresponding to the height s1 on the two sides) 
        x_s1_l = min(x[np.where(P >= P_s1)[0]])
        x_s1_r = max(x[np.where(P >= P_s1)[0]])

        # calculation of condifence edges (values of x corresponding to the height s2 on the two sides) 
        x_s2_l = min(x[np.where(P >= P_s2)[0]])
        x_s2_r = max(x[np.where(P >= P_s2)[0]])

        return P_s1, P_s2, x_s1_l, x_s1_r, x_s2_l, x_s2_r

#post_file, outname = '../data/data_gr_9dim_abhi.dat', 'fig1_gr_9dim_abhi'
#post_file, outname = '../data/data_mod_gr.txt', 'fig1_modgr'

snr=50

loc_2d = '/home/ajit.mehta/gr_consistency_highermodes/runs/modGR_simulations/SNR%d/NSBH_M120_iota1.57/'%snr

Nbins = 401

color = ['c', 'k', '#0072b2', '#CC79A7']

#dmc_2d, dq_2d, dmc_1d, dq_1d = np.loadtxt(post_file, unpack=True)

post_data = np.loadtxt('%s/emcee_samples_2d_and_1d.dat'%loc_2d, unpack=True)
mc0_2d, q0_2d, mc10_2d, q10_2d, q0_1d, q10_1d, mc0_1d, mc10_1d = post_data[:,0], post_data[:,1], post_data[:,2], post_data[:,3], post_data[:,4], post_data[:,5], post_data[:,6], post_data[:,7]

nn=5000
mc_2d=mc0_2d[nn:]
mc1_2d=mc10_2d[nn:]
q_2d=q0_2d[nn:]
q1_2d=q10_2d[nn:]
q_1d=q0_1d[nn:]
q1_1d=q10_1d[nn:]
mc_1d=mc0_1d[nn:]
mc1_1d=mc10_1d[nn:]

dmc_2d=mc_2d-mc1_2d
dq_2d=q_2d-q1_2d
dmc_1d=mc_1d-mc1_1d
dq_1d=q_1d-q1_1d
#dmc_bins = np.linspace(-0.07, 0.04, Nbins)
#dq_bins = np.linspace(-0.0008, 0.0004, Nbins)

dmc_bins = np.linspace(-0.6, 1.1, Nbins)
dq_bins = np.linspace(-0.004, 0.006, Nbins)
## for plots
dmc_min = -0.1
dmc_max = 0.5
dq_min = -0.0005
dq_max = 0.0025

dmc = np.mean(np.diff(dmc_bins))
dq = np.mean(np.diff(dq_bins))

dmc_intp = (dmc_bins[:-1] + dmc_bins[1:])/2.
dq_intp = (dq_bins[:-1] + dq_bins[1:])/2.

P_dmc_dq_2d, dmc_bins, dq_bins = np.histogram2d(dmc_2d, dq_2d, bins=(dmc_bins, dq_bins), normed=True)
P_dmc_dq_2d = P_dmc_dq_2d.T
P_dmc_marg = np.sum(P_dmc_dq_2d, axis=0) * dq
P_dq_marg = np.sum(P_dmc_dq_2d, axis=1) * dmc

P_dmc_1d, dmc_bins = np.histogram(dmc_1d,bins=dmc_bins, normed=True)
P_dq_1d, dq_bins = np.histogram(dq_1d,bins=dq_bins, normed=True)

s1_2d = ba.nsigma_value(P_dmc_dq_2d, 0.5)
s2_2d = ba.nsigma_value(P_dmc_dq_2d, 0.9)

s1_1d_v1, s2_1d_v1, left1_1d_v1, right1_1d_v1, left2_1d_v1, right2_1d_v1 = calc_cred_intervals_in_1d(P_dmc_1d, dmc_intp)
s1_1d_v2, s2_1d_v2, left1_1d_v2, right1_1d_v2, left2_1d_v2, right2_1d_v2 = calc_cred_intervals_in_1d(P_dq_1d, dq_intp)

s1_marg_v1, s2_marg_v1, left1_marg_v1, right1_marg_v1, left2_marg_v1, right2_marg_v1 = calc_cred_intervals_in_1d(P_dmc_marg, dmc_intp)
s1_marg_v2, s2_marg_v2, left1_marg_v2, right1_marg_v2, left2_marg_v2, right2_marg_v2 = calc_cred_intervals_in_1d(P_dq_marg, dq_intp)

p = plt.figure(figsize=(5,5))
ax1 = plt.subplot2grid((3,3), (0,0), colspan=2)
ax2 = plt.subplot2grid((3,3), (1,2), rowspan=2)
ax3 = plt.subplot2grid((3,3), (1,0), colspan=2, rowspan=2)

ax1.plot(dmc_intp, tgr.gf(P_dmc_1d),color=color[0], lw=1.5)
ax1.plot(dmc_intp, tgr.gf(P_dmc_marg),color=color[1], lw=1.5)
ax1.axvline(0.,color='k', ls='-', lw=0.5)
ax1.axvline(x=left2_1d_v1, color=color[0], ls=':', lw=0.5)
ax1.axvline(x=right2_1d_v1, color=color[0], ls=':', lw=0.5)
ax1.axvline(x=left2_marg_v1, color=color[1], ls=':', lw=0.5)
ax1.axvline(x=right2_marg_v1, color=color[1], ls=':', lw=0.5)
ax1.set_ylabel('$P(\Delta M_c)$',fontsize=14)
#ax1.set_xticks(np.arange(min(dmc_bins), max(dmc_bins)+0.1, 0.1))
#ax1.set_xticklabels(np.arange(min(dmc_bins), max(dmc_bins)+0.1, 0.1), fontsize=12)
ax1.set_xlim(dmc_min,dmc_max)
ax1.set_ylim(0,np.max(P_dmc_1d)*1.001)
ax1.xaxis.tick_top()
ax1.set_yticks([])

ax3.pcolormesh(dmc_bins, dq_bins, tgr.gf(P_dmc_dq_2d), cmap='RdPu')
ax3.contour(dmc_intp, dq_intp, tgr.gf(P_dmc_dq_2d), levels=(s2_2d,s1_2d), linewidths=(1,1.5), colors=color[1])
ax3.plot(0, 0, '+', color='white', ms=11, mew=2) # for mod gr make the marker k+ , for gr make the marker w+
ax3.set_xlabel('$\Delta M_c ~ (M_\odot)$',fontsize=14, labelpad=10)
ax3.set_ylabel('$\Delta q \, \\times \, 10^{3}$',fontsize=14)
ax3.set_xticks([])
ax3.set_yticks([])
ax3.set_xlim(dmc_min,dmc_max)
ax3.set_ylim(dq_min,dq_max)

ax2.plot(tgr.gf(P_dq_1d), dq_intp,color=color[0], lw=1.5)
ax2.plot(tgr.gf(P_dq_marg), dq_intp,color=color[1], lw=1.5)
ax2.axhline(0.,color='k', ls='-', lw=0.5)
ax2.axhline(y=left2_1d_v2, color=color[0], ls=':', lw=0.5)
ax2.axhline(y=right2_1d_v2, color=color[0], ls=':', lw=0.5)
ax2.axhline(y=left2_marg_v2, color=color[1], ls=':', lw=0.5)
ax2.axhline(y=right2_marg_v2, color=color[1], ls=':', lw=0.5)
ax2.set_xlabel('$P(\Delta q)$',fontsize=14, labelpad=10)
ax2.set_xticks([])

#if post_file == '../data/data_mod_gr.txt':
#ax2.set_yticks(np.arange(-0.0015, 0.0015, 0.0005))
#ax2.set_yticklabels(np.arange(-1.5, 1.5, 0.5), fontsize=12)
#ax2.set_ylim(-1.5e-3,5e-4)
#else: 
#ax2.set_yticks(np.arange(min(dq_bins), max(dq_bins)+0.0005, 0.0005))
#ax2.set_yticklabels(np.arange(min(dq_bins)*10**3, (max(dq_bins)+0.5)*10**3, 0.5), fontsize=12)
ax2.set_ylim(dq_min,dq_max)

ax2.set_xlim(0,np.max(P_dq_1d)*1.001)
ax2.yaxis.tick_right()
plt.tight_layout()
#plt.savefig('../papers/intro_paper/figs/%s.png'%outname, dpi=300)
plt.savefig('paper.pdf')
plt.close()
