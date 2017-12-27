from numpy import sqrt, sin, cos, pi
import matplotlib
matplotlib.use("Pdf")
import numpy as np
import matplotlib.pyplot as plt
import emcee
import template_hm as phhsi
from pycbc  import  detector
from lal import MSUN_SI, MTSUN_SI, PC_SI, PI, PC_SI, C_SI, GAMMA, MRSUN_SI




def lnlike(theta,dr,di,f1,psd):
	M,q,dL,iota,t0,phase=theta	
	f,hpf,hcf=phhsi.phenomhh_waveform_SI(M,q,dL,iota,t0,phase,f1[200],0.1)
	ra=1.
	dec =1.
	pol=0.
	Fp,Fc = detector.overhead_antenna_pattern(ra, dec, pol)	
	signal=Fp*hpf+Fc*hcf
	signalr=np.real(signal)
	signali=np.imag(signal)
	like_list=-((dr[200:9998]-signalr[200:9998])*(dr[200:9998]-signalr[200:9998])+(di[200:9998]-signali[200:9998])*(di[200:9998]-signali[200:9998]))/(psd[200:9998]) #likelihood list	
	like= 0.2*np.sum(like_list)#0.2 = 0.5*4.*df from the inner-product definition
	return like#log-likelihood



def lnprior(theta):
	M,q,dL,iota,t0,phase = theta
	if 1 < M < 200 and 0.097 < q <= 1. and 1.<dL<10000 and 0. <= iota <= pi and 0.<= t0 <= 15. and 0. <= phase <= 2.*pi:
		return 0.0
	return -np.inf



def lnprob(theta):
	lp = lnprior(theta)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(theta,dr,di,f1,psd)


##########################################################
###################### MAIN ##############################
##########################################################

loc='/home/siddharth.dhanpal/Work/projects/imrtestgr_hh/runs/201708_pe_results_phenomhh/m_70_q_5_i_90/22_plus_HM/data'
result=[21.4140366758,1./5,198.717477328,pi/2,6.,pi]#initial guess around which walkers start. This is also true value.Mc,q,dL,i,t0,initial_phase 


data=np.genfromtxt('%s/detected_data.txt'%loc,delimiter=' ')#df=0.1 for my data

di=data[:,2]  #real part of data
dr=data[:,1]  #imag part of data
f1=data[:,0]  #frequency 
psd=data[:,3] #power spectral density


ndim, nwalkers = 6, 100
pos = [result + np.array([0.002*result[0]*np.random.random_sample()-0.001*result[0],0.002*result[1]*np.random.random_sample()-0.001*result[1],0.002*result[2]*np.random.random_sample()-0.001*result[2],0.002*result[3]*np.random.random_sample()-0.001*result[3],0.002*result[4]*np.random.random_sample()-0.001*result[4],0.002*result[5]*np.random.random_sample()-0.001*result[5]]) for i in range(nwalkers)]



sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,threads=10)#threads--> no. of cores 

f = open("%s/chain_file.dat"%loc, "w")
f.close()
#writing data for each itteration.
for result in sampler.sample(pos, iterations=6000, storechain=False):
	position = result[0]
	f = open("%s/chain_file.dat"%loc, "a")
	for k in range(position.shape[0]):
		p=position[k]
		f.write("{0:1d} {1:2f} {2:3f} {3:4f} {4:5f} {5:6f} {6:7f}\n".format(k,p[0],p[1],p[2],p[3],p[4],p[5]))
	f.close()

