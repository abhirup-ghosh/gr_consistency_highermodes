'''
PE Code for the 'no-hair' test of BHs with an amplitude correction 'c1' in the Higher Modes of the waveform

'''
# --------------------------------------------------------- # 

import os, sys
from numpy import sqrt, sin, cos, pi
import matplotlib
matplotlib.use("Pdf")
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import emcee
sys.path.append('/home/abhirup/Documents/Work/gr_consistency_highermodes/src')
#sys.path.append('/home/tousif.islam/gr_consistency_highermodes_old/src')
import template_tgr as phhsi
from pycbc  import  detector
from lal import MSUN_SI, MTSUN_SI, PC_SI, PI, PC_SI, C_SI, GAMMA, MRSUN_SI
import corner
from optparse import OptionParser
import time

# -------------------- likelihod -------------------------- # 

def lnlike(param_vec, data1, data2, data3, freq1, psd, psdv, f_low, f_cut):
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
        df = np.diff(freq1)[0]

        N_low=np.int((f_low-freq1[0])/df)
        N_cut=np.int((f_cut-freq1[0])/df)

        Nls=np.int(f_low/df)  #N_low_signal
        Ncs=np.int(f_cut/df)  #N_cut_signal

        # unpacking the parameter vector 
        Mc, q, c, dL, i, t0, phi0, ra, sin_dec, pol= param_vec
        dec=np.arcsin(sin_dec)
        # three detectors
        H = detector.Detector("H1")
        L = detector.Detector("L1")
        V = detector.Detector("V1")
        # time of arrival/ time delay in detectors wrt earth center
        dt01=H.time_delay_from_earth_center(ra, dec, t0)
        dt02=L.time_delay_from_earth_center(ra, dec, t0)
        dt03=V.time_delay_from_earth_center(ra, dec, t0)
        t01=t0+dt01
        t02=t0+dt02
        t03=t0+dt03
        # generate the waveforms [time shifted] 
        f1, hpf1, hcf1 = phhsi.phenomhh_waveform_real_ampcorr_SI(Mc, q, c, c, c, dL, i, t01, (phi0 %(2.*pi)), f_low, df, Ncs)
        f2, hpf2, hcf2 = phhsi.phenomhh_waveform_real_ampcorr_SI(Mc, q, c, c, c, dL, i, t02, (phi0 %(2.*pi)), f_low, df, Ncs)
        f3, hpf3, hcf3 = phhsi.phenomhh_waveform_real_ampcorr_SI(Mc, q, c, c, c, dL, i, t03, (phi0 %(2.*pi)), f_low, df, Ncs)

        # compute antenna patterns 
        Fp1,Fc1 = H.antenna_pattern(ra, dec, pol, t0)
        Fp2,Fc2 = L.antenna_pattern(ra, dec, pol, t0)
        Fp3,Fc3 = V.antenna_pattern(ra, dec, pol, t0)

        # signal in the detectors
        signal1=Fp1*hpf1 + Fc1*hcf1
        signal2=Fp2*hpf2 + Fc2*hcf2
        signal3=Fp3*hpf3 + Fc3*hcf3

        like1 = -2.*df*np.real(np.dot(data1[N_low:N_cut]-signal1[Nls:Ncs],np.conj((data1[N_low:N_cut]-signal1[Nls:Ncs])/psd[N_low:N_cut])))
        like2 = -2.*df*np.real(np.dot(data2[N_low:N_cut]-signal2[Nls:Ncs],np.conj((data2[N_low:N_cut]-signal2[Nls:Ncs])/psd[N_low:N_cut])))
        like3 = -2.*df*np.real(np.dot(data3[N_low:N_cut]-signal3[Nls:Ncs],np.conj((data3[N_low:N_cut]-signal3[Nls:Ncs])/psdv[N_low:N_cut])))
        like=like1+like2+like3

        return like#log-likelihood


# -------------------- priors -------------------------- # 

def lnprior(param_vec):
	Mc, q, c, dL, i, t0, phi_0, ra, sin_dec, pol = param_vec
	if 1 < Mc < 200 and 0.05 < q <= 1. and  -10.0 < c <= 10.0  and 1.<dL<10000 and 0.<= i <= pi and 0.<= t0 <= 15. and -pi <= phi_0 <= 3.*pi and 0. <= ra < 2.*pi and -1. <= sin_dec <= 1. and 0. <= pol <= pi:
		return 2.*np.log(dL)+np.log(np.sin(i))  
	return -np.inf

# -------------------- total probability -------------------------- # 

def lnprob(param_vec):
        lp = lnprior(param_vec)
        if not np.isfinite(lp):
                return -np.inf
        return lp + lnlike(param_vec, data1, data2, data3, freq1, psd, psdv, f_low, f_cut)


##########################################################
###################### MAIN ##############################
##########################################################

start_time = time.time()

# -------------------- inputs -------------------------- # 
parser = OptionParser()
parser.add_option("-H", "--data-fname-H1", dest="data_fname_H1", help="data filename : H1")
parser.add_option("-L", "--data-fname-L1", dest="data_fname_L1", help="data filename : L1")
parser.add_option("-V", "--data-fname-V1", dest="data_fname_V1", help="data filename : V1")
parser.add_option("-o", "--out-dir", dest="out_dir", help="output directory")
parser.add_option("-i", "--init-loc", dest="init_loc", help="location for initial conditions")

(options, args) = parser.parse_args()
data_fname_H1 = options.data_fname_H1
data_fname_L1 = options.data_fname_L1
data_fname_V1 = options.data_fname_V1
out_dir = options.out_dir
init_loc = options.init_loc

result = np.loadtxt(init_loc, unpack=True)

os.system('mkdir -p %s'%out_dir)
os.system('cp -r %s %s'%(data_fname_H1, out_dir))
os.system('cp -r %s %s'%(data_fname_L1, out_dir))
os.system('cp -r %s %s'%(data_fname_V1, out_dir))
os.system('cp -r %s %s'%(init_loc, out_dir))
os.system('cp %s %s' %(__file__, out_dir))


f_low = 20.
f_cut = 999.

ndim, nwalkers = 10, 200
num_threads = 30
num_iter = 20000
# ------------------------------------------------------ # 


# read the detector data in Fourier domain. [fourier freq, real part of the data, imaginary part of the data, psd]
# read the detector data in Fourier domain. [fourier freq, real part of the data, imaginary part of the data, psd]
freq1, dr1, di1, psd = np.loadtxt(data_fname_H1, unpack=True)
data1 = dr1 + 1j*di1
print '... read data from H1 detector'
freq2, dr2, di2, psd = np.loadtxt(data_fname_L1, unpack=True)
data2 = dr2 + 1j*di2
print '... read data from L1 detector'
freq3, dr3, di3, psdv = np.loadtxt(data_fname_V1, unpack=True)
data3 = dr3 + 1j*di3
print '... read data from V1 detector'

#################################################################
# emcee runs
#################################################################


# create initial walkers

mc_init, q_init, c_init, dL_init, iota_init, t0_init, phi0_init, ra_init, sin_dec_init, pol_init = result

pos = [result + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

print '... generated initial walkers. starting sampling...' 

# sample the likelihood using EMCEE 
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=num_threads)
sampler.run_mcmc(pos, num_iter)

mc_chain, q_chain, c_chain, dL_chain, iota_chain, t0_chain, phi0_chain, ra_chain, sin_dec_chain, pol_chain = sampler.chain[:, :, 0].T, sampler.chain[:, :, 1].T, sampler.chain[:, :, 2].T, sampler.chain[:, :, 3].T, sampler.chain[:, :, 4].T, sampler.chain[:, :, 5].T, sampler.chain[:, :, 6].T, sampler.chain[:, :, 7].T, sampler.chain[:, :, 8].T, sampler.chain[:, :, 9].T

samples = sampler.chain[:, :, :].reshape((-1, ndim))

#################################################################
# plotting and saving data
#################################################################

# save the data
np.savetxt(out_dir+'/emcee_samples.dat', samples, header='mc q c1 dL i t0 phi0 ra sin(dec) pol')

# plot the data and the psd 
df = np.mean(np.diff(freq1))
idx = np.logical_and(freq1 > 20, freq1 < 999)
snr1 = 2*np.sqrt(df*np.sum(abs(data1[idx])**2/psd[idx]))
snr2 = 2*np.sqrt(df*np.sum(abs(data2[idx])**2/psd[idx]))
snr3 = 2*np.sqrt(df*np.sum(abs(data3[idx])**2/psdv[idx]))
snr=np.sqrt(snr1*snr1+snr2*snr2+snr3*snr3)

plt.figure(figsize=(8,6))
plt.loglog(freq1, abs(data1), 'r',label='H1')
plt.loglog(freq2, abs(data2), 'b',label='L1')
plt.loglog(freq3, abs(data3), 'g',label='V1')
plt.loglog(freq1, psd**0.5, 'c')
plt.xlim(20,1e3)
plt.ylim(1e-24,5e-23)
plt.xlabel('$f$ [Hz]')
plt.ylabel('$h(f)$ and $S_h(f)$')
plt.title('snr = %2.1f' %snr)
plt.legend()
plt.savefig('%s/data.png'%out_dir, dpi=200)

print '... plotted data'

#c1_init=absc_init*np.exp(1j*phi_init)
#c1_init=abs(c1_init)
#c1_chain=cr_chain+1j*cc_chain
#c1_chain=abs(c1_chain)

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
plt.plot(c_chain, color="k", alpha=0.4, lw=0.5)
plt.plot(c_init + np.std(c_chain, axis=1), 'r')
plt.axhline(y=c_init, color='g')
plt.ylabel('c1')
plt.subplot(624)
plt.plot(dL_chain, color="k", alpha=0.4, lw=0.5)
plt.plot(dL_init + np.std(dL_chain, axis=1), 'r')
plt.axhline(y=dL_init, color='g')
plt.ylabel('dL')
plt.subplot(625)
plt.plot(iota_chain, color="k", alpha=0.4, lw=0.5)
plt.plot(iota_init + np.std(iota_chain, axis=1), 'r')
plt.axhline(y=iota_init, color='g')
plt.ylabel('iota')
plt.subplot(626)
plt.plot(t0_chain, color="k", alpha=0.4, lw=0.5)
plt.plot(t0_init + np.std(t0_chain, axis=1), 'r')
plt.axhline(y=t0_init, color='g')
plt.ylabel('t0')
plt.subplot(627)
plt.plot(phi0_chain, color="k", alpha=0.4, lw=0.5)
plt.plot(phi0_init + np.std(phi0_chain, axis=1), 'r')
plt.axhline(y=phi0_init, color='g')
plt.ylabel('phi0')
plt.subplot(6,2,8)
plt.plot(ra_chain, color="k", alpha=0.4, lw=0.5)
plt.plot(ra_init + np.std(ra_chain, axis=1), 'r')
plt.axhline(y=ra_init, color='g')
plt.ylabel('ra')
plt.subplot(6,2,9)
plt.plot(sin_dec_chain, color="k", alpha=0.4, lw=0.5)
plt.plot(sin_dec_init + np.std(sin_dec_chain, axis=1), 'r')
plt.axhline(y=sin_dec_init, color='g')
plt.ylabel('dec')
plt.subplot(6,2,10)
plt.plot(pol_chain, color="k", alpha=0.4, lw=0.5)
plt.plot(pol_init + np.std(pol_chain, axis=1), 'r')
plt.axhline(y=pol_init, color='g')
plt.ylabel('pol')
plt.savefig('%s/samples_chain.png'%out_dir, dpi=300)

# corner plots
plt.figure()
corner.corner(samples, labels=['mc', 'q', 'c1', 'dL', 'i', 't0', 'phi0', 'ra', 'sin(dec)', 'pol'])
plt.savefig("%s/corner_plot_wo_burnin.png"%out_dir)
plt.close()

print '... plotted corner plot' 

end_time = time.time()
print '... time taken: %.2f seconds'%(end_time-start_time)

