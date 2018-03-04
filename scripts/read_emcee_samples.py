import numpy as np

def read_emcee_samples(post_loc, nwalkers, num_iter, ndim, n_burnin):

	samples_preburnin = np.loadtxt(post_loc).reshape(nwalkers, num_iter, ndim)
	samples_postburnin = samples_preburnin[:,n_burnin:,:].reshape(-1,ndim)
	mc, q, mc1, q1, dL, i, t0, phi0, ra, sin_dec, pol = samples_postburnin[:,0],samples_postburnin[:,1],samples_postburnin[:,2],samples_postburnin[:,3],samples_postburnin[:,4],samples_postburnin[:,5],samples_postburnin[:,6],samples_postburnin[:,7],samples_postburnin[:,8],samples_postburnin[:,9],samples_postburnin[:,10]
	return mc, q, mc1, q1, dL, i, t0, phi0, ra, sin_dec, pol
