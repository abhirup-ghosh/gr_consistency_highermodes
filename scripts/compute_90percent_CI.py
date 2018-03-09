import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import bayesian as ba
import read_emcee_samples as res
import standard_gwtransf as gw

def calc_conf_intervals_in_1d(P, x):

        # find the value of P corresponding to 68% and 95% confidence heights 
        P_s1 = ba.nsigma_value(P, 0.5)
        P_s2 = ba.nsigma_value(P, 0.9)

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

in_dir_root = '../runs/9_param_runs/Mc_q_deltaMc_diffM'
burnin = 1000

M_inj_list = [20,40,60,80,100,120,140,160,180,200]#40
q_inj = 9#[1,1.5,2,3,4,5,6,7,8,9]#9
iota_inj_list = [30, 45, 60, 80, 90]

out_matrix = np.zeros([len(M_inj_list), len(iota_inj_list)+1])

for (i, M_inj) in enumerate(M_inj_list):
  for (j, iota_inj) in enumerate(iota_inj_list):

    #if q_inj == 1.5:
    #	post_loc = in_dir_root + '/M_40_q_1.5_iota_%d/emcee_samples.dat'%(iota_inj)
    #else:
    post_loc = in_dir_root + '/M_%d_q_9_iota_%d/emcee_samples.dat'%(M_inj, iota_inj)
    nwalkers, num_iter, ndim = 100, 3000, 11
    n_burnin = 1000
    mc, q, mc1, q1, dL, iota, t0, phi0, ra, sin_dec, pol = res.read_emcee_samples(post_loc, nwalkers, num_iter, ndim, n_burnin)
    x, x1 = mc, mc1    

    eta_inj = gw.eta_from_q(q_inj)
    mc_inj = gw.mc_from_toteta(M_inj, eta_inj)
    dxbyx_1d = (x[burnin:] - x1[burnin:])/mc_inj

    Nbins = 101
    dxbyx_bins = np.linspace(min(dxbyx_1d), max(dxbyx_1d), Nbins)
    dxbyx = np.mean(np.diff(dxbyx_bins))
    dxbyx_intp = (dxbyx_bins[:-1] + dxbyx_bins[1:])/2.

    P_dxbyx_1d, dxbyx_bins = np.histogram(dxbyx_1d,bins=dxbyx_bins, normed=True)

    s1_1d, s2_1d, left1_1d, right1_1d, left2_1d, right2_1d = calc_conf_intervals_in_1d(P_dxbyx_1d, dxbyx_intp)

    out_matrix[i, 0] = M_inj
    out_matrix[i, j+1] = right2_1d-left2_1d

np.savetxt('../data/90_percent_width_9dim_DeltaMcbyMcinj_diffM_abhi.txt', out_matrix, header='M i_30 i_45 i_60 i_80 i_90')
