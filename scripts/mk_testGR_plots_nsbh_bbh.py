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
import scipy
from scipy import interpolate
import scipy.ndimage.filters as filter


""" gaussian filter of histogram """
def gf(P):
	return filter.gaussian_filter(P, sigma=2.0)


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

# make the 2D and marginalized 1d posteriors of delta_Mc and delta_q 
def mk_posterior_plot(post_data, col, Nbins, linest, lab): 

	# read posterior samples 
	post_data = np.loadtxt('%s/emcee_samples_2d_and_1d.dat'%loc, unpack=True)
	mc0_2d, q0_2d, mc10_2d, q10_2d, q0_1d, q10_1d, mc0_1d, mc10_1d = post_data[:,0], post_data[:,1], post_data[:,2], post_data[:,3], post_data[:,4], post_data[:,5], post_data[:,6], post_data[:,7]

	# remove the burn-in part 
	nn=5000
	mc_2d=mc0_2d[nn:]
	mc1_2d=mc10_2d[nn:]
	q_2d=q0_2d[nn:]
	q1_2d=q10_2d[nn:]
	q_1d=q0_1d[nn:]
	q1_1d=q10_1d[nn:]
	mc_1d=mc0_1d[nn:]
	mc1_1d=mc10_1d[nn:]

	# calculate Delta_mc and delta_q 
	dmc_2d=mc_2d-mc1_2d
	dq_2d=q_2d-q1_2d
	dmc_1d=mc_1d-mc1_1d
	dq_1d=q_1d-q1_1d

	# create bins to evaluate the histogram 
	dmc_bins = np.linspace(-0.6, 1.1, Nbins)
	dq_bins = np.linspace(-0.004, 0.006, Nbins)


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


	ax1.plot(dmc_intp, gf(P_dmc_marg), color=col, lw=1.5, ls=linest)
	ax1.axvline(0.,color='k', ls='-', lw=0.5)
	ax1.set_ylabel('$P(\Delta M_c)$',fontsize=14)
	ax1.set_xlim(dmc_min,dmc_max)
	ax1.set_ylim(0,np.max(P_dmc_marg)*1.001)
	ax1.xaxis.tick_top()
	ax1.set_yticks([])

	#ax3.pcolormesh(dmc_bins, dq_bins, gf(P_dmc_dq_2d), cmap='RdPu')
	ax3.contour(dmc_intp, dq_intp, gf(P_dmc_dq_2d), levels=(s2_2d,s1_2d), linewidths=(1,1.5), colors=col, linestyles=linest, label=lab)
	ax3.plot(0, 0, '+', color='k', ms=11, mew=2) # for mod gr make the marker k+ , for gr make the marker w+
	ax3.set_xlabel('$\Delta M_c ~ (M_\odot)$',fontsize=14, labelpad=10)
	ax3.set_ylabel('$\Delta q \, \\times \, 10^{3}$',fontsize=14)
	ax3.set_xticks([])
	ax3.set_yticks([])
	ax3.set_xlim(dmc_min,dmc_max)
	ax3.set_ylim(dq_min,dq_max)

	ax2.plot(gf(P_dq_marg), dq_intp*1e3, color=col, lw=1.5, ls=linest)
	ax2.axhline(0., color='k', ls='-', lw=0.5)
	ax2.set_xlabel('$P(\Delta q)$',fontsize=14, labelpad=10)
	ax2.set_xticks([])
	ax2.set_ylim(dq_min*1e3, dq_max*1e3)

	ax2.set_xlim(0,np.max(P_dq_marg)*1.001)
	ax2.yaxis.tick_right()

	return plt 

## MAIN CODE 
snr=50
Nbins = 401

col_v = ['orange', 'k']
linest_v = ['-', '-']
lab_v = ['BBH', 'non-BBH']

plt.figure(figsize=(5,5))
ax1 = plt.subplot2grid((3,3), (0,0), colspan=2)
ax2 = plt.subplot2grid((3,3), (1,2), rowspan=2)
ax3 = plt.subplot2grid((3,3), (1,0), colspan=2, rowspan=2)

# axis limits for plots
dmc_min = -0.2
dmc_max = 0.5
dq_min = -0.001
dq_max = 0.0025

for i, system in enumerate(['BBH', 'NSBH']): 
	
	linest = linest_v[i]
	col = col_v[i]
	lab = lab_v[i]

	loc = '../runs/modGR_simulations/SNR%d/%s_M120_iota1.57/'%(snr, system)
	print loc, linest 
	plt = mk_posterior_plot(loc, col, Nbins, linest, lab) 

plt.legend(frameon=False)
plt.tight_layout()
plt.savefig('posteriors_BBH_rescaled_NSBH.pdf')
