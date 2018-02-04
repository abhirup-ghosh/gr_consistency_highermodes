import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import bayesian as ba

def calc_conf_intervals_in_1d(P, x):

        # find the value of P corresponding to 68% and 95% confidence heights 
        P_s1 = ba.nsigma_value(P, 0.68)
        P_s2 = ba.nsigma_value(P, 0.95)

        # calculation of condifence edges (values of x corresponding to the height s1 on the two sides) 
        x_s1_l = min(x[np.where(P >= P_s1)[0]])
        x_s1_r = max(x[np.where(P >= P_s1)[0]])

        # calculation of condifence edges (values of x corresponding to the height s2 on the two sides) 
        x_s2_l = min(x[np.where(P >= P_s2)[0]])
        x_s2_r = max(x[np.where(P >= P_s2)[0]])

        return P_s1, P_s2, x_s1_l, x_s1_r, x_s2_l, x_s2_r

##################################################
# MAIN
##################################################

in_dir_root = '../runs/Mc_q_deltaMc'
burnin = 1000

M_inj = 40
q_inj_list = [2,3,4,5,6,7,8,9]
iota_inj_list = [30, 45, 60, 80, 90]

for q_inj in q_inj_list:
  for iota_inj in iota_inj_list:
    post_loc = in_dir_root + '/M_40_q_%d_iota_%d/emcee_samples.dat'%(q_inj, iota_inj)
    mc, q, mc1, q1, dL, iota, t0, phi0 = np.loadtxt(post_loc, unpack=True)
    
    dmcbymc_1d = (mc[burnin:] - mc1[burnin:])/mc[burnin:]

    Nbins = 1601
    dmcbymc_bins = np.linspace(-20.0, 20.0, Nbins)
    dmcbymc = np.mean(np.diff(dmcbymc_bins))
    dmcbymc_intp = (dmcbymc_bins[:-1] + dmcbymc_bins[1:])/2.

    P_dmcbymc_1d, dmcbymc_bins = np.histogram(dmcbymc_1d,bins=dmcbymc_bins, normed=True)

    s1_1d_v1, s2_1d_v1, left1_1d_v1, right1_1d_v1, left2_1d_v1, right2_1d_v1 = calc_conf_intervals_in_1d(P_dmcbymc_1d, dmcbymc_intp)

    print '90 percent CI for M=40, q=%d, iota=%d: %.4f'%(q_inj, iota_inj, right2_1d_v1-left2_1d_v1) 
