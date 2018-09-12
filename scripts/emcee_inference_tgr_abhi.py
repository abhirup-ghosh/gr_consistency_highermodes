import os, sys
from numpy import sqrt, sin, cos, pi
import matplotlib
matplotlib.use("Pdf")
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import emcee
sys.path.insert(0, '/home/abhirup/Documents/Work/gr_consistency_highermodes/src')
import template_tgr as phhsi
from pycbc  import  detector
from lal import MSUN_SI, MTSUN_SI, PC_SI, PI, PC_SI, C_SI, GAMMA, MRSUN_SI
import corner
from optparse import OptionParser
import time


def lnlike(param_vec, data, freq, psd, f_low, f_cut):
	"""
	compute the log likelihood
	
	inputs: 
	param_vec : vector of parameters 
	dr, di, 
	freq : Fourier freq 
	psd : psd vector 
	flow,fcut
	
	output: 
	log_likelhood 
	"""
	df = np.diff(freq)[0]#np.mean(np.diff(freq))

        N_low=np.int((f_low-freq[0])/df)
        N_cut=np.int((f_cut-freq[0])/df)

        Nls=np.int(f_low/df)  #N_low_signal
        Ncs=np.int(f_cut/df)  #N_cut_signal

	# unpacking the parameter vector 
	Mc, q, Mc1, q1, dL, i, t0, phi0,  ra, sin_dec, pol= param_vec

	# generate the waveform 
	f, hpf, hcf = phhsi.phenomhh_waveform_SI(Mc, q, Mc1, q, dL, i, t0, (phi0 %(2.*pi)), f_low, df, Ncs)

	# compute antenna patterns 
	Fp,Fc = detector.overhead_antenna_pattern(ra, np.arcsin(sin_dec), pol)

	signal=Fp*hpf+Fc*hcf

	like = -2.*df*np.real(np.dot(data[N_low:N_cut]-signal[Nls:Ncs],np.conj((data[N_low:N_cut]-signal[Nls:Ncs])/psd[N_low:N_cut])))

	return like#log-likelihood


def lnprior(param_vec):
	Mc, q, Mc1, q1, dL, i, t0, phi_0, ra, sin_dec, pol = param_vec
	if 1 < Mc < 200 and 0.05 < q <= 1. and  1 < Mc1 < 200 and 0.05 < q1 <= 1. and 1.<dL<10000 and 0.<= i <= pi and 0.<= t0 <= 15. and -pi <= phi_0 <= 3.*pi and 0. <= ra < 2.*pi and -1. <= sin_dec <= 1. and -pi <= pol <= 0.:
		return 2.*np.log(dL)+np.log(np.sin(i))
	return -np.inf



def lnprob(param_vec):
	lp = lnprior(param_vec)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(param_vec, data, freq, psd, f_low, f_cut)


##########################################################
###################### MAIN ##############################
##########################################################

start_time = time.time()

# -------------------- inputs -------------------------- # 
parser = OptionParser()
parser.add_option("-d", "--data-fname", dest="data_fname", help="data filename")
parser.add_option("-o", "--out-dir", dest="out_dir", help="output directory")
parser.add_option("-i", "--init-cond", dest="result", help="initial conditions")
(options, args) = parser.parse_args()
data_fname = options.data_fname
out_dir = options.out_dir
result = options.result
result = map(float, result.strip('[]').split(','))

os.system('mkdir -p %s'%out_dir)
os.system('cp -r %s %s'%(data_fname, out_dir))
os.system('cp %s %s' %(__file__, out_dir))

f_low = 20.
f_cut = 4096.

ndim, nwalkers = 11, 100
num_threads = 32
num_iter = 2000
# ------------------------------------------------------ # 


# read the detector data in Fourier domain. [fourier freq, real part of the data, imaginary part of the data, psd]
freq, dr, di, psd = np.loadtxt(data_fname, unpack=True)
data = dr + 1j*di 
print '... read data' 

# create initial walkers

mc_init, q_init, mc1_init, q1_init, dL_init, iota_init, t0_init, phi0_init, ra_init, sin_dec_init, pol_init = result

pos = [result + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

print '... generated initial walkers. starting sampling...' 

# sample the likelihood using EMCEE 
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=num_threads)
sampler.run_mcmc(pos, num_iter)

mc_chain, q_chain, mc1_chain, q1_chain, dL_chain, iota_chain, t0_chain, phi0_chain, ra_chain, sin_dec_chain, pol_chain = sampler.chain[:, :, 0].T, sampler.chain[:, :, 1].T, sampler.chain[:, :, 2].T, sampler.chain[:, :, 3].T, sampler.chain[:, :, 4].T, sampler.chain[:, :, 5].T, sampler.chain[:, :, 6].T, sampler.chain[:, :, 7].T, sampler.chain[:, :, 8].T, sampler.chain[:, :, 9].T, sampler.chain[:, :, 10].T

samples = sampler.chain[:, :, :].reshape((-1, ndim))

#################################################################
# plotting and saving data
#################################################################

# save the data
np.savetxt(out_dir+'/emcee_samples.dat', samples, header='mc q mc1 q1 dL i t0 phi0 ra sin(dec) pol')

# plot the data and the psd 
df = np.mean(np.diff(freq))
idx = np.logical_and(freq > 20, freq < 999)
snr = 2*np.sqrt(df*np.sum(abs(data[idx])**2/psd[idx]))

plt.figure(figsize=(8,6))
plt.loglog(freq, abs(data), 'r')
plt.loglog(freq, psd**0.5, 'c')
plt.xlim(20,1e3)
plt.ylim(1e-24,5e-23)
plt.xlabel('$f$ [Hz]')
plt.ylabel('$h(f)$ and $S_h(f)$')
plt.title('snr = %2.1f' %snr)
plt.savefig('%s/data.png'%out_dir, dpi=200)

print '... plotted data'

# Inspiral Chain plot
plt.figure(figsize=(16,8))
plt.subplot(621)
plt.plot(mc_chain, color="k", alpha=0.4, lw=0.5)
plt.plot(mc_init + np.std(mc_chain, axis=1), 'r')
plt.axhline(y=mc_init, color='g')
plt.ylabel('mc')
plt.subplot(622)
plt.plot(q_chain, color="k", alpha=0.4, lw=0.5)
plt.plot(q_init + np.std(q_chain, axis=1), 'r')
plt.axhline(y=q_init, color='g')
plt.ylabel('q')
plt.subplot(623)
plt.plot(mc1_chain, color="k", alpha=0.4, lw=0.5)
plt.plot(mc1_init + np.std(mc1_chain, axis=1), 'r')
plt.axhline(y=mc1_init, color='g')
plt.ylabel('mc1')
plt.subplot(624)
plt.plot(q1_chain, color="k", alpha=0.4, lw=0.5)
plt.plot(q1_init + np.std(q1_chain, axis=1), 'r')
plt.axhline(y=q1_init, color='g')
plt.ylabel('q1')
plt.subplot(625)
plt.plot(dL_chain, color="k", alpha=0.4, lw=0.5)
plt.plot(dL_init + np.std(dL_chain, axis=1), 'r')
plt.axhline(y=dL_init, color='g')
plt.ylabel('dL')
plt.subplot(626)
plt.plot(iota_chain, color="k", alpha=0.4, lw=0.5)
plt.plot(iota_init + np.std(iota_chain, axis=1), 'r')
plt.axhline(y=iota_init, color='g')
plt.ylabel('iota')
plt.subplot(627)
plt.plot(t0_chain, color="k", alpha=0.4, lw=0.5)
plt.plot(t0_init + np.std(t0_chain, axis=1), 'r')
plt.axhline(y=t0_init, color='g')
plt.ylabel('t0')
plt.subplot(628)
plt.plot(phi0_chain, color="k", alpha=0.4, lw=0.5)
plt.plot(phi0_init + np.std(phi0_chain, axis=1), 'r')
plt.axhline(y=phi0_init, color='g')
plt.ylabel('phi0')
plt.subplot(629)
plt.plot(ra_chain, color="k", alpha=0.4, lw=0.5)
plt.plot(ra_init + np.std(ra_chain, axis=1), 'r')
plt.axhline(y=ra_init, color='g')
plt.ylabel('ra')
plt.subplot(6,2,10)
plt.plot(sin_dec_chain, color="k", alpha=0.4, lw=0.5)
plt.plot(sin_dec_init + np.std(sin_dec_chain, axis=1), 'r')
plt.axhline(y=sin_dec_init, color='g')
plt.ylabel('dec')
plt.subplot(6,2,11)
plt.plot(pol_chain, color="k", alpha=0.4, lw=0.5)
plt.plot(pol_init + np.std(pol_chain, axis=1), 'r')
plt.axhline(y=pol_init, color='g')
plt.ylabel('pol')
plt.savefig(out_dir + '/samples_chain.png', dpi=300)

# corner plots
plt.figure()
corner.corner(samples, labels=['mc', 'q', 'mc1', 'q1', 'dL', 'i', 't0', 'phi0', 'ra', 'sin(dec)', 'pol'])
plt.savefig("%s/corner_plot_wo_burnin.png"%out_dir)
plt.close()

print '... plotted corner plot' 

end_time = time.time()
print '... time taken: %.2f seconds'%(end_time-start_time)

