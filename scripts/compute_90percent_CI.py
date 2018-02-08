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

in_dir_root = '../runs/Mc_q_deltaq'
burnin = 1000

M_inj = 40
q_inj_list = [2,3,4,5,6,7,8,9]
iota_inj_list = [30, 45, 60, 80, 90]

out_matrix = np.zeros([len(q_inj_list), len(iota_inj_list)+1])

for (i, q_inj) in enumerate(q_inj_list):
  for (j, iota_inj) in enumerate(iota_inj_list):

    post_loc = in_dir_root + '/M_40_q_%d_iota_%d/emcee_samples.dat'%(q_inj, iota_inj)
    mc, q, mc1, q1, dL, iota, t0, phi0 = np.loadtxt(post_loc, unpack=True)
    x, x1 = q, q1    

    dxbyx_1d = (x[burnin:] - x1[burnin:])/x[burnin:]

    Nbins = 101
    dxbyx_bins = np.linspace(min(dxbyx_1d), max(dxbyx_1d), Nbins)
    dxbyx = np.mean(np.diff(dxbyx_bins))
    dxbyx_intp = (dxbyx_bins[:-1] + dxbyx_bins[1:])/2.

    P_dxbyx_1d, dxbyx_bins = np.histogram(dxbyx_1d,bins=dxbyx_bins, normed=True)

    s1_1d, s2_1d, left1_1d, right1_1d, left2_1d, right2_1d = calc_conf_intervals_in_1d(P_dxbyx_1d, dxbyx_intp)

    out_matrix[i, 0] = q_inj
    out_matrix[i, j+1] = right2_1d-left2_1d

np.savetxt('../data/90_percent_width_dq_t_abhi.txt', out_matrix, header='q i_30 i_45 i_60 i_80 i_90')
