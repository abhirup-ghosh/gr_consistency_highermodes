from numpy import sqrt, sin, cos, pi
import matplotlib
matplotlib.use("Pdf")
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import emcee
import template_hm as phhsi
from pycbc  import  detector
from lal import MSUN_SI, MTSUN_SI, PC_SI, PI, PC_SI, C_SI, GAMMA, MRSUN_SI
import corner




def lnlike(param_vec, dr, di, freq, psd, f_low, f_cut):
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
	df = np.mean(np.diff(freq))

        N_low=np.int((f_low-freq[0])/df)
        N_cut=np.int((f_cut-freq[0])/df)

        Nls=np.int(f_low/df)  #N_low_signal
        Ncs=np.int(f_cut/df)  #N_cut_signal
	
	# unpacking the parameter vector 
	Mc, q, dL, ci, t0, phi_0 = param_vec	

	# generate the waveform 
	f, hpf, hcf = phhsi.phenomhh_waveform_SI(Mc, q, dL, np.arccos(ci), t0, phi_0, flow, df, Ncs)

	ra=1.
	dec =1.
	pol=0.
	# compute antenna patterns 
	Fp,Fc = detector.overhead_antenna_pattern(ra, dec, pol)	

	signal=Fp*hpf+Fc*hcf

	like = -2.*df*np.real(np.dot(data[N_low:N_cut]-signal[Nls:Ncs],np.conj((data[N_low:N_cut]-signal[Nls:Ncs])/psd[N_low:N_cut])))

	return like#log-likelihood


def lnprior(param_vec):
	Mc, q, dL, ci, t0, phi_0 = param_vec
	if 1 < Mc < 200 and 0.05 < q <= 1. and 1.<dL<10000 and -1. < ci <= 1. and 0.<= t0 <= 15. and 0. <= phi_0 <= 2.*pi:
		return 2.*np.log(dL)
	return -np.inf



def lnprob(param_vec):
	lp = lnprior(param_vec)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(param_vec,dr,di,freq,psd)


##########################################################
###################### MAIN ##############################
##########################################################


# -------------------- inputs -------------------------- # 
loc='/home/siddharth.dhanpal/Work/projects/imrtestgr_hh/runs/201708_pe_results_phenomhh/m_70_q_5_i_90/22_plus_HM/data'
loc = '/home/ajith/working/cbc/gr_consistency_highermodes/runs/201708_pe_results_phenomhh/m_70_q_5_i_90/22_plus_HM/data'
loc = '/home/siddharth.dhanpal/Work/projects/imrtestgr_hh/runs/experiments_201709/M_80/q_9/i_60/data'
loc = '/home/ajith/working/cbc/gr_consistency_highermodes/runs/experiments_201709/M_80/q_9/i_60/data'

result=[21.4140366758,1./5,198.717477328,np.cos(pi/2),6.,pi]#initial guess around which walkers start. This is also true value.Mc,q,dL,i,t0,initial_phase 
data_fname = 'detected_data.txt'

# labels of the parameter vector 
param_label = ['$M_c$', '$q$', '$dL$', 'cos($\iota$)', '$t_0$', '$\phi_0$']


ndim, nwalkers = 6, 100
num_threads = 24 
num_iter = 100 
# ------------------------------------------------------ # 


# read the detector data in Fourier domain. [fourier freq, real part of the data, imaginary part of the data, psd]
freq, dr, di, psd = np.loadtxt(loc+'/'+data_fname, unpack=True)
data = dr + 1j*di 
print '... read data' 

# plot the data and the psd 
df = np.mean(np.diff(freq))
idx = np.logical_and(freq > 20, freq < 500)
snr = 2*np.sqrt(df*np.sum(abs(data[idx])**2/psd[idx]))

plt.figure(figsize=(8,6))
plt.loglog(freq, abs(data), 'r')
plt.loglog(freq, psd**0.5, 'c')
plt.xlim(20,1e3)
plt.ylim(1e-24,5e-23)
plt.xlabel('$f$ [Hz]')
plt.ylabel('$h(f)$ and $S_h(f)$')
plt.title('snr = %2.1f' %snr)
plt.savefig('%s/data.png'%loc, dpi=200)

print '... plotted data' 

# create initial walkers 
pos = [result + np.array([0.002*result[0]*np.random.random_sample()-0.001*result[0],0.002*result[1]*np.random.random_sample()-0.001*result[1],0.002*result[2]*np.random.random_sample()-0.001*result[2],0.002*result[3]*np.random.random_sample()-0.001*result[3],0.002*result[4]*np.random.random_sample()-0.001*result[4],0.002*result[5]*np.random.random_sample()-0.001*result[5]]) for i in range(nwalkers)]

# make corner plot of the initial walkers 
plt.figure()
figure = corner.corner(pos, labels=param_label, bins=10)
figure.tight_layout()
figure.savefig('%s/initial_walkers_corner.png' %loc, dpi=200)    

print '... generated initial walkers. starting sampling...' 

# sample the likelihood using EMCEE 
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=num_threads)

# writing data for each itteration.
for result in sampler.sample(pos, iterations=num_iter, storechain=False):
	position = result[0]
	f = open("%s/chain_file.dat"%loc, "a")
	for k in range(position.shape[0]):
		p=position[k]
		f.write("{0:1d} {1:2f} {2:3f} {3:4f} {4:5f} {5:6f} {6:7f}\n".format(k,p[0],p[1],p[2],p[3],p[4],p[5]))
	f.close()

# add the final corner plots here
data=np.genfromtxt('%s/chain_file.dat'%loc,delimiter=' ')

chain_length=int(len(data[:,0])/n_walkers)
s=(nwalkers,chain_length,ndim)
x=np.zeros(s)

i=0
for j in range(x.shape[1]):
	for l in range(x.shape[0]):
		for k in range(x.shape[2]):
			x[l][j][k]=data[i][k+1]
		i+=1

samples = x[:,:,:].reshape((-1, ndim))

plt.figure()
corner.corner(samples, labels=param_label)
plt.savefig("%s/final_corner_plot_without_burnin.png"%loc_plots)
plt.close()
 
