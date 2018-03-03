from numpy import sqrt, sin, cos, pi
import matplotlib
matplotlib.use("Pdf")
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import emcee
import template_tgr as phhsi
from pycbc  import  detector
from lal import MSUN_SI, MTSUN_SI, PC_SI, PI, PC_SI, C_SI, GAMMA, MRSUN_SI
import corner




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
	df = np.mean(np.diff(freq))

        N_low=np.int((f_low-freq[0])/df)
        N_cut=np.int((f_cut-freq[0])/df)

        Nls=np.int(f_low/df)  #N_low_signal
        Ncs=np.int(f_cut/df)  #N_cut_signal
	
	# unpacking the parameter vector 
	Mc, q, Mc1, q1, dL, i, t0, phi_0, ra, sin_dec, pol= param_vec	

	# generate the waveform 
	f, hpf, hcf = phhsi.phenomhh_waveform_SI(Mc, q, Mc1, q1, dL, i, t0, (phi_0 %(2.*pi)), f_low, df, Ncs)

	# compute antenna patterns 
	Fp,Fc = detector.overhead_antenna_pattern(ra, np.arcsin(sin_dec), pol)	

	signal=Fp*hpf+Fc*hcf

	like = -2.*df*np.real(np.dot(data[N_low:N_cut]-signal[Nls:Ncs],np.conj((data[N_low:N_cut]-signal[Nls:Ncs])/psd[N_low:N_cut])))

	return like#log-likelihood


def lnprior(param_vec):
	Mc, q, Mc1, q1, dL, i, t0, phi_0, ra, sin_dec, pol = param_vec
	if 1 < Mc < 200 and 0.05 < q <= 1. and  1 < Mc1 < 200 and 0.05 < q1 <= 1. and 1.<dL<10000 and 0.<= i <= pi and 0.<= t0 <= 15. and -pi <= phi_0 <= 3.*pi and 0. <= ra < 2.*pi and -1. <= sin_dec <= 1. and 0. <= pol <= pi:
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


# -------------------- inputs -------------------------- # 
loc = '/home/siddharth.dhanpal/Work/projects/imrtestgr_hh/scripts/final_scripts'#'/home/siddharth.dhanpal/Work/projects/imrtestgr_hh/runs/201711_pe_deviations_noise_free_analysis/M_80/q_9/i_60/data'

result=[ 1.886407405431894801e+01, 1./9, 1.886407405431894801e+01, 1./9, 489.747597, pi/3, 6., pi, 1., np.sin(1.), 0.01]#initial guess around which walkers start. This is also true value.Mc,q,dL,i,t0,initial_phase for quicker convergence. *If any injection value is 0 add 0.01 to it. 
data_fname = 'detected_data.txt'

# labels of the parameter vector 
param_label = ['$M_c$', '$q$', '$M_c_1$', '$q_1$' ,'$dL$', '$\iota$', '$t_0$', '$\phi_0$', '$ra$', '$sin(dec)$', '$\Psi$']

f_low = 20.
f_cut = 999.

ndim, nwalkers = 11, 100
num_threads = 24 
num_iter = 3000 
# ------------------------------------------------------ # 


# read the detector data in Fourier domain. [fourier freq, real part of the data, imaginary part of the data, psd]
freq, dr, di, psd = np.loadtxt(loc+'/'+data_fname, unpack=True)
data = dr + 1j*di 
print '... read data' 

# create initial walkers 
pos = [result + np.array([0.0002*result[0]*np.random.random_sample()-0.0001*result[0],0.002*result[1]*np.random.random_sample()-0.001*result[1],0.0002*result[2]*np.random.random_sample()-0.0001*result[2],0.002*result[3]*np.random.random_sample()-0.001*result[3],0.002*result[4]*np.random.random_sample()-0.001*result[4],0.002*result[5]*np.random.random_sample()-0.001*result[5],0.0002*result[6]*np.random.random_sample()-0.0001*result[6],0.002*result[7]*np.random.random_sample()-0.001*result[7],0.002*result[8]*np.random.random_sample()-0.001*result[8],0.002*result[9]*np.random.random_sample()-0.001*result[9],0.002*result[10]*np.random.random_sample()-0.001*result[10]]) for i in range(nwalkers)]

##make corner plot of the initial walkers 
#plt.figure()
#figure = corner.corner(pos, labels=param_label, bins=10)
#figure.tight_layout()
#figure.savefig('%s/initial_walkers_corner.png' %loc, dpi=200)    

print '... generated initial walkers. starting sampling...' 

# sample the likelihood using EMCEE 
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=num_threads)

# writing data for each itteration.
for result in sampler.sample(pos, iterations=num_iter, storechain=False):
	position = result[0]
	f = open("%s/chain_tgr_2_var.dat"%loc, "a")
	for k in range(position.shape[0]):
		p=position[k]
                f.write("{0:1d} {1:2f} {2:3f} {3:4f} {4:5f} {5:6f} {6:7f} {7:8f} {8:8f} {9:8f} {10:8f} {11:8f}\n".format(k,p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7]%(2.*pi),p[8]%(2.*pi),p[9]%(2.*pi),p[10]%(2.*pi)))# Order: walker number, Mc, q, Mc1, q1, dL, iota, t0, phi_0, ra, dec, pol
	f.close()



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
plt.savefig('%s/data.png'%loc, dpi=200)

print '... plotted data'

# add the final corner plots here
chain_data=np.genfromtxt('%s/chain_tgr_2_var.dat'%loc,delimiter=' ')

chain_length=int(len(chain_data[:,0])/n_walkers)
s=(nwalkers,chain_length,ndim)
x=np.zeros(s)

i=0
for j in range(x.shape[1]):
	for l in range(x.shape[0]):
		for k in range(x.shape[2]):
			x[l][j][k]=chain_data[i][k+1]
		i+=1

samples = x[:,:,:].reshape((-1, ndim))


plt.figure()
corner.corner(samples, labels=param_label)
plt.savefig("%s/final_corner_plot_without_burnin.png"%loc)
plt.close()

print '... plotted corner plot' 

