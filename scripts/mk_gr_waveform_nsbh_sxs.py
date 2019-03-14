import matplotlib as mpl
mpl.use('Agg')
import sys
sys.path.append('../src/')
import pycbc
import pycbc.filter.matchedfilter as mfilter
import pycbc.psd
import pycbc.noise.gaussian
from pycbc  import  detector
import numpy as np
import matplotlib.pyplot as plt
import lal
import glob
import os
import scipy
from scipy import interpolate
from scipy.signal import argrelextrema
import template_hm as phhsi
from lal import MSUN_SI, MTSUN_SI, PC_SI, PI, PC_SI, C_SI, GAMMA, MRSUN_SI
import time

""" taper time domain data. h is a numpy array (Ref. Eq. (3.35) of gr-qc/0001023) """
def taper_waveform(h):
        h_temp = h
        peakind = np.array(argrelextrema(abs(h_temp), np.greater)).flatten()
        idx_peak2 = peakind[1]          # index of second extremum
        startind = np.flatnonzero(h_temp)[0]            # index of first non-zero data point

        # taper from start to second extremum 
        n = idx_peak2 - startind
        # do the taper using formula Eq. (3.35) of gr-qc/0001023.
        h_temp[startind] = 0
        for i in range(startind+1, startind+n-2):
                z = (n - 1.)/(i-startind) + (n-1.)/(i-startind - (n-1.))
                h_temp[i] = h_temp[i]*1./(np.exp(z) + 1)
        return h_temp


# signal parameters
f_low=20.
srate = 2048.

m1, m2 = 60.0, 10.0
M=m1+m2          # in MSUN
q=m2/m1  
eta=m1*m2/M**2.       
Mc=(M*q**0.6)/((1.+q)**1.2)
SNR_req=200.    
iota_list=[0.00, 0.79, 1.05, 1.57]

phi0 = 1.3
psi_list = [0.00, -1.57]

ra=0.          
dec =0.
pol=0.

cbc_list = ['BBH','NSBH']

data_dir = '/home/ajit.mehta/Ajit_work/phenom_hh/data/polarizations/four_modes'
out_dir = '/home/abhirup/Documents/Work/gr_consistency_highermodes/injections/SXS_four_modes_20190114_SNR_200'
#out_dir = '/home/ajit.mehta/gr_consistency_highermodes/injections/SXS_four_modes'
#out_dir = '/home/ajit.mehta/gr_consistency_highermodes/test'

for cbc in cbc_list:
  for iota in iota_list:
    for psi in psi_list:
	start_time = time.time()

	out_file = '%s_M_%.2f_iota_%.2f_psi_%.2f_t0_0'%(cbc, M, iota, psi)

	print "... case:", cbc, iota, psi
        print '...with Mtot = %.2f'%M

	# reading data
        data_loc = data_dir + '/NRPolzns_%s_SpEC_q6.00_spin1[0.00,0.00,-0.00]_spin2[-0.00,-0.00,-0.00]_iota_%.2f_psi_%.2f.npz'%(cbc, iota, psi)
        data = np.load(data_loc)
        t_geom = data['t']
        hp_geom = data['hp']
        hc_geom = data['hc']
        print "... read data"

	r = 1. #in Mpc

	# converting from geometric to SI units for distance r = 1 Mpc
	# and computing the evolution of phase phi(t) and instantaneous frequency F(t) = (1/2*pi)*(dphi/dt)
	t_SI = t_geom * (M*lal.MTSUN_SI)
	hp_SI = hp_geom*(M*lal.MTSUN_SI)/(r*1e6*lal.PC_SI/lal.C_SI)
	hc_SI = hc_geom*(M*lal.MTSUN_SI)/(r*1e6*lal.PC_SI/lal.C_SI)
	h_SI = hp_SI + 1j*hc_SI

	# restricting waveform to  lower range in order to compute FFT.
        idx_rstrctd = np.arange(len(t_SI)-300000,len(t_SI),1)
	t_SI_rstrctd, hp_SI_rstrctd, hc_SI_rstrctd = t_SI[idx_rstrctd], hp_SI[idx_rstrctd], hc_SI[idx_rstrctd]

	# defining t0 as the time at ISCO, and redefining it as 0
        t0 = (5./256.)*M*lal.MTSUN_SI/((np.pi*M*lal.MTSUN_SI*f_low)**(8./3.)*eta)
        print '... t0 (time corresponding to ISCO of initial (non-spinning) binary): %.4f seconds'%t0
        t_SI_rstrctd = t_SI_rstrctd - t_SI_rstrctd[0] - t0
        t0 = 0.

	dt_SI_rstrctd = np.diff(t_SI_rstrctd)[0]

        ## Note : There is no need to do interpolation here, as the data that I have provided is already uniformally sampled. Also, it's not good idea to             
	##        interpolate complicated oscillating functions like we have for higher modes waveform.

	# computing SNR for r = 1Mpc
	N = len(hp_SI_rstrctd)
	f_SI = np.fft.fftfreq(N, d=dt_SI_rstrctd)
	df = np.diff(f_SI)[0]
	print '... df: %f'%df
	Fp,Fc = detector.overhead_antenna_pattern(ra, dec, pol)
        psd = pycbc.psd.aLIGOZeroDetHighPower(N, df, f_low)
	
	signal=Fp*hp_SI_rstrctd+Fc*hc_SI_rstrctd
	signal_time=pycbc.types.timeseries.TimeSeries(signal,delta_t=dt_SI_rstrctd,dtype=float)
        SNR=mfilter.sigma(signal_time,psd=psd,low_frequency_cutoff=f_low)

	print '... initial SNR:%f'%SNR

	# rescaling (distance, hp, hc) for fixed SNR = 25
	r = SNR/SNR_req
        hp_SI_rescaled = hp_SI_rstrctd/r
        hc_SI_rescaled = hc_SI_rstrctd/r

	# sanity check: recomputing SNR for rescaled waveform (confirm SNR = 25)
	signal=Fp*hp_SI_rescaled+Fc*hc_SI_rescaled
        signal_time=pycbc.types.timeseries.TimeSeries(signal,delta_t=dt_SI_rstrctd,dtype=float)
        SNR=mfilter.sigma(signal_time,psd=psd,low_frequency_cutoff=f_low)

	print '... rescaled distance: %f Mpc for a fixed SNR: %f'%(r, SNR)

	# generate Fourier domain waveform
	signal_freq = np.fft.fft(taper_waveform(signal))*dt_SI_rstrctd
	data_long=signal_freq#+noise ## comment noise to generate noise free data

        Psi_ref=phi0 + psi
        t0=0.  
        incl_angle = iota
        f, hpf, hcf = phhsi.phenomhh_waveform_SI(Mc,q,r,incl_angle,t0,Psi_ref,f_low,df,int(N/2.+1))
        NN = len(f)        
        data_long = data_long[0:NN]
        psd = psd[0:NN]
        best_fit_signal=Fp*hpf+Fc*hcf

        ## data is non-zero only for f>=f_low like phenomhh waveform.
        data = np.zeros_like(f, dtype=np.complex128)
        band_idx = f >= (f_low)
        data[band_idx]=data_long[band_idx]


        ## Fitting to get t0 starts here....
        f_isco = 2200./M
        fmax_fit=6*f_isco
        #t0=0.075
        phase_data = np.unwrap(np.angle(data))
        phase_tmp = np.unwrap(np.angle(best_fit_signal))#-2*np.pi*f*t0
        delta_phase = phase_tmp - phase_data
        index = np.where((f>f_low)&(f<fmax_fit))
        fit_coef = np.polyfit(2*np.pi*f[index],delta_phase[index],1)
        t0 = fit_coef[0]
        print '...printing fit coeffs...'
        print fit_coef[0], fit_coef[1]
        phase_tmp = phase_tmp - (2*np.pi*f*fit_coef[0] + fit_coef[1])
        delta_phase_fit = 2*np.pi*f*fit_coef[0] + fit_coef[1]

        fmax=1000.

	# plotting Fourier data and PSD
	plt.figure(figsize=(12,6))
        plt.subplot(231)
	plt.loglog(f, psd**0.5, 'c')
        plt.loglog(f, abs(data),'r',lw=2, label='SXS')
        plt.loglog(f, abs(best_fit_signal), 'k',lw=2,alpha=0.7,label='phenomhm')
	plt.xlim([f_low, fmax])
        plt.ylim([5e-27, 5e-23])
        plt.xlabel('f')
        plt.ylabel('$\\tilde{h}(f)$')
        plt.legend(loc='best')
        plt.title('%s, four modes with inclination, $\iota=%.2f$'%(cbc,incl_angle))
        plt.subplot(232)
        plt.semilogx(f,phase_data,label='SXS')
        plt.semilogx(f,phase_tmp,ls='--',label='phenom')
        plt.ylim([0, 800.])
        plt.xlim([f_low, fmax])
        plt.xlabel('f')
	plt.ylabel('$\phi$')
        plt.legend(loc='best')
        plt.subplot(234)
        plt.semilogx(f[index],phase_data[index]-phase_tmp[index])
        #plt.ylim(-10,10)
        plt.xlabel('f')
	plt.ylabel('$\phi_{data} - \phi_{template}$')
        plt.subplot(233)
        plt.plot(f[index],delta_phase[index],lw=2)
        plt.plot(f[index],delta_phase_fit[index],'--')
        plt.ylabel('$\Delta\phi_{data,fit}$')
        plt.xlabel('f')
	plt.subplot(235)
	plt.loglog(f, np.real(data), ls='solid', color='k', label='data',lw=0.1)
	plt.loglog(f, np.real(best_fit_signal), ls='dashed', color='r', label='best fit signal',lw=0.1)
	plt.legend(loc='best')
	plt.ylim([5e-27, 5e-23])
	plt.xlim([f_low, fmax])
	plt.subplot(236)
        plt.loglog(f, np.imag(data), ls='solid', color='k', label='data',lw=0.1)
        plt.loglog(f, np.imag(best_fit_signal), ls='dashed', color='r', label='best fit signal',lw=0.1)
        plt.legend(loc='best')
        plt.ylim([5e-27, 5e-23])
        plt.xlim([f_low, fmax])
        plt.tight_layout()
        plt.savefig(out_dir + '/%s_data.png'%out_file)
	plt.close()

	# saving data
	np.savetxt(out_dir + '/%s_sxs_data.dat'%out_file, np.c_[f, np.real(data), np.imag(data) ,psd], header='f real_data imag_data psd')
	np.savetxt(out_dir + '/%s_phenomhm_data.dat'%out_file, np.c_[f, np.real(best_fit_signal), np.imag(best_fit_signal) ,psd], header='f real_data imag_data psd')
	np.savetxt(out_dir + '/%s_initial.dat'%out_file, np.c_[Mc, q, r, iota, t0, Psi_ref, ra, np.sin(dec), pol], header='Mc q r iota t0 Psi_ref ra sin_dec pol', fmt=['%.4f','%.4f','%.4f','%.4f','%.4f','%.4f','%.4f','%.4f','%.4f'])
	end_time = time.time()
	print "... time taken: %.2f seconds"%(end_time-start_time)
