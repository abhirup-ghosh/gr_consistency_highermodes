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

parser = OptionParser()
parser.add_option("-d", "--post-loc", dest="post_loc", help="path to data folder")
parser.add_option("-n", "--Nbins", dest="Nbins", help="Nbins for the binning")
(options, args) = parser.parse_args()
post_loc = options.post_loc
Nbins = options.Nbins

color = ['#0072b2', '#CC79A7']

dmc_2d, dq_2d, dmc_1d, dq_1d = np.loadtxt(post_loc, unpack=True)

dmc_bins = np.linspace(-0.2, 0.2, Nbins)
dq_bins = np.linspace(-0.002, 0.001, Nbins)
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

s1_2d = ba.nsigma_value(P_dmc_dq_2d, 0.68)
s2_2d = ba.nsigma_value(P_dmc_dq_2d, 0.95)

s1_1d_v1, s2_1d_v1, left1_1d_v1, right1_1d_v1, left2_1d_v1, right2_1d_v1 = calc_cred_intervals_in_1d(P_dmc_1d, dmc_intp)
s1_1d_v2, s2_1d_v2, left1_1d_v2, right1_1d_v2, left2_1d_v2, right2_1d_v2 = calc_cred_intervals_in_1d(P_dq_1d, dq_intp)

s1_marg_v1, s2_marg_v1, left1_marg_v1, right1_marg_v1, left2_marg_v1, right2_marg_v1 = calc_cred_intervals_in_1d(P_dmc_marg, dmc_intp)
s1_marg_v2, s2_marg_v2, left1_marg_v2, right1_marg_v2, left2_marg_v2, right2_marg_v2 = calc_cred_intervals_in_1d(P_dq_marg, dq_intp)

p = plt.figure(figsize=(5,5))
ax1 = plt.subplot2grid((3,3), (0,0), colspan=2)
ax2 = plt.subplot2grid((3,3), (1,2), rowspan=2)
ax3 = plt.subplot2grid((3,3), (1,0), colspan=2, rowspan=2)

ax1.plot(dmc_intp, tgr.gf(P_dmc_1d),color=color[0], lw=1)
ax1.plot(dmc_intp, tgr.gf(P_dmc_marg),color=color[1], lw=1)
ax1.axvline(0.,color='k')
#ax1.axvline(x=left1_1d_v1, color='b', ls='-.')
#ax1.axvline(x=right1_1d_v1, color='b', ls='-.')
ax1.axvline(x=left2_1d_v1, color=color[0], ls='--')
ax1.axvline(x=right2_1d_v1, color=color[0], ls='--')
#ax1.axvline(x=left1_marg_v1, color='k', ls='-.')
#ax1.axvline(x=right1_marg_v1, color='k', ls='-.')
ax1.axvline(x=left2_marg_v1, color=color[1], ls='--')
ax1.axvline(x=right2_marg_v1, color=color[1], ls='--')
ax1.set_ylabel('$P(\Delta M_c)$',fontsize=14)
ax1.set_xticks(np.arange(-0.2, 0.2, 0.1))
ax1.xaxis.tick_top()
ax1.set_yticks([])

#ax3.pcolormesh(dmc_bins, dq_bins, tgr.gf(P_dmc_dq_2d), cmap='YlOrBr')
ax3.contour(dmc_intp, dq_intp, tgr.gf(P_dmc_dq_2d), levels=(s2_2d,s1_2d), linewidths=(1,1.5), colors=color[0])
ax3.plot(0, 0, 'k+', ms=12, mew=2) # for mod gr make the marker k+ , for gr make the marker w+
ax3.set_xlabel('$\Delta M_c (M_\odot)$',fontsize=14, labelpad=10)
ax3.set_ylabel('$\Delta q$',fontsize=14)
ax3.set_xticks([])
ax3.set_yticks([])

ax2.plot(tgr.gf(P_dq_1d), dq_intp,color=color[0], lw=1)
ax2.plot(tgr.gf(P_dq_marg), dq_intp,color=color[1], lw=1)
ax2.axhline(0.,color='k')
#ax2.axhline(y=left1_1d_v2, color='b', ls='-.')
#ax2.axhline(y=right1_1d_v2, color='b', ls='-.')
ax2.axhline(y=left2_1d_v2, color=color[0], ls='--')
ax2.axhline(y=right2_1d_v2, color=color[0], ls='--')
#ax2.axhline(y=left1_marg_v2, color='k', ls='-.')
#ax2.axhline(y=right1_marg_v2, color='k', ls='-.')
ax2.axhline(y=left2_marg_v2, color=color[1], ls='--')
ax2.axhline(y=right2_marg_v2, color=color[1], ls='--')
ax2.set_xlabel('$P(\Delta q)$',fontsize=14, labelpad=10)
ax2.set_xticks([])
ax2.set_yticks(np.arange(-0.002, 0.001, 0.0005))
ax2.yaxis.tick_right()
plt.tight_layout()
plt.savefig('fig1_modGR.png', dpi=300)
plt.savefig('fig1_modGR.pdf')
plt.show()
