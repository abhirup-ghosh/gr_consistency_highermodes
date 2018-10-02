import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import corner
import read_emcee_samples as res


post_loc = '../runs/9_param_runs/Mc_q_deltaMc_deltaq/M_80_q_9_iota_60/emcee_samples.dat'

nwalkers, num_iter, ndim, n_burnin = 100, 3000, 11, 1000
samples_preburnin = np.loadtxt(post_loc).reshape(nwalkers, num_iter, ndim)
samples_postburnin = samples_preburnin[:,n_burnin:,:].reshape(-1,ndim)

mc, q, mc1, q1, dL, iota, t0, phi0, ra, sin_dec, pol = res.read_emcee_samples(post_loc, nwalkers, num_iter, ndim, n_burnin)
dmc = mc1 - mc
dq = q1 - q
samples_postburnin = np.column_stack((mc, q, dmc, dq, dL, iota, t0, phi0, ra, sin_dec, pol))

plt.figure()
corner.corner(samples_postburnin, labels=['$M_c$', 'q', '$\Delta M_c$', '$\Delta q$', 'dL', 'i', 't0', 'phi0', 'ra', 'sin(dec)', 'pol'])
plt.savefig("../plots/corner_plot_postburnin_Mc_q_deltaMc_deltaq_M_80_q_9_iota_60.png")
plt.close()
