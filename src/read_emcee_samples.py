import numpy as np

def read_emcee_samples_11dim(post_loc, nwalkers, num_iter, ndim, n_burnin):

	samples_preburnin = np.loadtxt(post_loc).reshape(nwalkers, num_iter, ndim)
	samples_postburnin = samples_preburnin[:,n_burnin:,:].reshape(-1,ndim)
	mc, q, mc1, q1, dL, i, t0, phi0, ra, sin_dec, pol = samples_postburnin[:,0],samples_postburnin[:,1],samples_postburnin[:,2],samples_postburnin[:,3],samples_postburnin[:,4],samples_postburnin[:,5],samples_postburnin[:,6],samples_postburnin[:,7],samples_postburnin[:,8],samples_postburnin[:,9],samples_postburnin[:,10]
	return mc, q, mc1, q1, dL, i, t0, phi0, ra, sin_dec, pol

def read_emcee_samples_9dim(post_loc, nwalkers, num_iter, ndim, n_burnin):

        samples_preburnin = np.loadtxt(post_loc).reshape(nwalkers, num_iter, ndim)
        samples_postburnin = samples_preburnin[:,n_burnin:,:].reshape(-1,ndim)
        mc, q, dL, i, t0, phi0, ra, sin_dec, pol = samples_postburnin[:,0],samples_postburnin[:,1],samples_postburnin[:,2],samples_postburnin[:,3],samples_postburnin[:,4],samples_postburnin[:,5],samples_postburnin[:,6],samples_postburnin[:,7],samples_postburnin[:,8]
        return mc, q, dL, i, t0, phi0, ra, sin_dec, pol


if __name__ == '__main__':
	post_loc_deltaMc_deltaq = '../runs/9_param_runs/Mc_q_deltaMc_deltaq/M_80_q_9_iota_60/emcee_samples.dat'
	post_loc_deltaMc = '../runs/9_param_runs/Mc_q_deltaMc_deltaq/M_80_q_9_iota_60_deltaMc/emcee_samples.dat'
	post_loc_deltaq = '../runs/9_param_runs/Mc_q_deltaMc_deltaq/M_80_q_9_iota_60_deltaq/emcee_samples.dat'

	nwalkers, num_iter, ndim = 100, 3000, 11
	n_burnin = 1000
	
	mc, q, mc1, q1, dL, i, t0, phi0, ra, sin_dec, pol = read_emcee_samples(post_loc_deltaMc_deltaq, nwalkers, num_iter, ndim, n_burnin)	
	delta_mc_2d, delta_q_2d = mc - mc1, q - q1

	mc, q, mc1, q1, dL, i, t0, phi0, ra, sin_dec, pol = read_emcee_samples(post_loc_deltaMc, nwalkers, num_iter, ndim, n_burnin)
        delta_mc_1d = mc - mc1

	mc, q, mc1, q1, dL, i, t0, phi0, ra, sin_dec, pol = read_emcee_samples(post_loc_deltaq, nwalkers, num_iter, ndim, n_burnin)
        delta_q_1d = q - q1

	np.savetxt('../data/data_gr_9dim_abhi.dat', np.c_[delta_mc_2d, delta_q_2d, delta_mc_1d, delta_q_1d], header='delta_mc_2d delta_q_2d delta_mc_1d delta_q_1d')
