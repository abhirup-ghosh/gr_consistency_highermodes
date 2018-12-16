import matplotlib as mpl
mpl.use('Agg')
import sys
sys.path.append('/home/ajit.mehta/Ajit_work/phenom_hh/src/')
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
import phenomhh_tgr as phh
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
<<<<<<< HEAD
flow =5.
f_low=20.
=======
flow=10.
>>>>>>> e867df712b1f99e394e0e97af4c518cd1e3e95ae
flow_snr=20.
srate = 2048.

m1, m2 = 60.0, 10.0
M=m1+m2          # in MSUN
q=m2/m1  
eta=m1*m2/M**2.       
Mc=(M*q**0.6)/((1.+q)**1.2)
SNR_req=25.    
<<<<<<< HEAD
iota_list=[1.57]#,0.79,1.57]
=======
iota_list=[0.00]#,0.79,1.57]
>>>>>>> e867df712b1f99e394e0e97af4c518cd1e3e95ae
Psi_ref=1.3

ra=0.          
dec =0.
pol_list=[0.00]#,-1.57,-3.14]

cbc_list = ['BBH']#''NSBH']

<<<<<<< HEAD
data_dir = '/home/ajit.mehta/Ajit_work/phenom_hh/data/polarizations/four_modes'
out_dir = '/home/ajit.mehta/gr_consistency_highermodes/plots/four_modes'
=======
data_dir = '/home/abhirup/Documents/Work/gr_consistency_highermodes/data/polarizations'
out_dir = '/home/abhirup/Documents/Work/gr_consistency_highermodes/injections/nsbh_sxs_20181209'
>>>>>>> e867df712b1f99e394e0e97af4c518cd1e3e95ae

for cbc in cbc_list:
  for iota in iota_list:
    for pol in pol_list:
	start_time = time.time()

	out_file = '%s_M_%.2f_iota_%.2f_pol_%.2f_t0_0'%(cbc, M, iota, pol)

	print "... case:", cbc, iota, pol
        print '...with Mtot = %.2f'%M

	# reading data
        data_loc = data_dir + '/NRPolzns_%s_SpEC_q6.00_spin1[0.00,0.00,-0.00]_spin2[-0.00,-0.00,-0.00]_iota_%.2f_psi_%.2f.npz'%(cbc, iota, pol)
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
	phi_SI = np.unwrap(np.angle(h_SI))
	Foft_SI = np.gradient(phi_SI)/np.gradient(t_SI)/(2*np.pi)

	# restricting waveform to  lower range in order to compute FFT.
        idx_rstrctd = np.arange(len(t_SI)-300000,len(t_SI),1)
	phi_SI_rstrctd, Foft_SI_rstrctd = phi_SI[idx_rstrctd], Foft_SI[idx_rstrctd]
	t_SI_rstrctd, hp_SI_rstrctd, hc_SI_rstrctd = t_SI[idx_rstrctd], hp_SI[idx_rstrctd], hc_SI[idx_rstrctd]

	# defining t0 as the time at ISCO, and redefining it as 0
        t0 = (5./256.)*M*lal.MTSUN_SI/((np.pi*M*lal.MTSUN_SI*flow)**(8./3.)*eta)
        print '... t0 (time corresponding to ISCO of initial (non-spinning) binary): %.4f seconds'%t0
        t_SI_rstrctd = t_SI_rstrctd - t_SI_rstrctd[0] - t0
        t0 = 0.

	# interpolating h_p(t) and h_c(t) over a equally-space time series with sampling rate 16kHz
	t_SI_rstrctd_interp = np.arange(t_SI_rstrctd[0], t_SI_rstrctd[-1], 1./srate)
	dt_SI_rstrctd_interp = np.diff(t_SI_rstrctd_interp)[0]
	print '... srate: %f, dt:%f'%(srate, dt_SI_rstrctd_interp)
        hp_SI_rstrctd_interp_obj = scipy.interpolate.interp1d(t_SI_rstrctd, hp_SI_rstrctd, fill_value=0., bounds_error=False)
        hc_SI_rstrctd_interp_obj = scipy.interpolate.interp1d(t_SI_rstrctd, hc_SI_rstrctd, fill_value=0., bounds_error=False)
        hp_SI_rstrctd_interp = hp_SI_rstrctd_interp_obj(t_SI_rstrctd_interp)
        hc_SI_rstrctd_interp = hc_SI_rstrctd_interp_obj(t_SI_rstrctd_interp)

	# plotting waveform before and after interpolation (and Foft_SI)
	plt.figure(figsize=(10,10))
        plt.subplot(511)
        plt.plot(t_SI_rstrctd, hp_SI_rstrctd, 'k', lw=0.5, label='before')
        plt.plot(t_SI_rstrctd_interp, hp_SI_rstrctd_interp, 'r--',lw=0.5, label='after')
	plt.legend(loc='best')
	plt.ylabel('$h_p$')
        plt.subplot(512)
        plt.plot(t_SI_rstrctd, hc_SI_rstrctd, 'k', lw=0.5, label='before')
        plt.plot(t_SI_rstrctd_interp, hc_SI_rstrctd_interp, 'r--',lw=0.5, label='after')
	plt.legend(loc='best')
	plt.ylabel('$h_c$')
	plt.subplot(513)
        plt.plot(t_SI_rstrctd, phi_SI_rstrctd, 'k', lw=0.2)
	plt.ylabel('$\phi(t)$')
	plt.subplot(514)
        plt.plot(t_SI_rstrctd, Foft_SI_rstrctd, 'k', lw=0.2)
	plt.axhline(y=15, color='r', ls='--')
	plt.ylabel('$F(t)$')
	plt.subplot(515)
        plt.plot(t_SI_rstrctd, Foft_SI_rstrctd, 'k', lw=0.2)
	plt.ylabel('$F(t)$')
	plt.xlim([-0.02, 0.02])
        plt.axhline(y=15, color='r', ls='--')
	plt.tight_layout()
	plt.savefig(out_dir + '/%s_interpolation.png'%out_file, dpi=300)
        plt.close()

	# computing SNR for r = 1Mpc
	N = len(hp_SI_rstrctd_interp)
	f_SI = np.fft.fftfreq(N, d=dt_SI_rstrctd_interp)
	df = np.diff(f_SI)[0]
	print '... df: %f'%df
	Fp,Fc = detector.overhead_antenna_pattern(ra, dec, pol)
        psd = pycbc.psd.aLIGOZeroDetHighPower(N, df, flow)
	
	signal=Fp*hp_SI_rstrctd_interp+Fc*hc_SI_rstrctd_interp
	signal_time=pycbc.types.timeseries.TimeSeries(signal,delta_t=dt_SI_rstrctd_interp,dtype=float)
        SNR=mfilter.sigma(signal_time,psd=psd,low_frequency_cutoff=flow_snr)

	print '... initial SNR:%f'%SNR

	# rescaling (distance, hp, hc) for fixed SNR = 25
	r = SNR/SNR_req
        hp_SI_rescaled = hp_SI_rstrctd_interp/r
        hc_SI_rescaled = hc_SI_rstrctd_interp/r

	# sanity check: recomputing SNR for rescaled waveform (confirm SNR = 25)
	signal=Fp*hp_SI_rescaled+Fc*hc_SI_rescaled
        signal_time=pycbc.types.timeseries.TimeSeries(signal,delta_t=dt_SI_rstrctd_interp,dtype=float)
        SNR=mfilter.sigma(signal_time,psd=psd,low_frequency_cutoff=flow_snr)

	print '... rescaled distance: %f Mpc for a fixed SNR: %f'%(r, SNR)

	# generate Fourier domain waveform
	signal_freq = np.fft.fft(taper_waveform(signal))*dt_SI_rstrctd_interp
	data=signal_freq#+noise ## comment noise to generate noise free data

        m1=m1*MSUN_SI
        m2=m2*MSUN_SI
        mt=m1+m2
        incl_angle = iota
        phi=0.
        lmax=4 
        hpf, hcf = phh.generate_phenomhmv1_fd(m1, m2, incl_angle, phi, f_low, df, int(N/2.+1), lmax, [[2,2],[2,1],[3,3],[4,4]], Psi_ref)
        NN = int(N/2.+1)
        f = np.linspace(0., df*(NN-1), NN)
        data = data[0:NN]
        psd = psd[0:NN]
        datar=np.real(data)
        datai=np.imag(data)

        hpf=hpf*mt*MRSUN_SI*MTSUN_SI*mt/(MSUN_SI*MSUN_SI*(1.0e6*r*PC_SI))
        hcf=hcf*mt*MRSUN_SI*MTSUN_SI*mt/(MSUN_SI*MSUN_SI*(1.0e6*r*PC_SI))
        best_fit_signal=Fp*hpf+Fc*hcf

	# plotting Fourier data and PSD
	plt.figure(figsize=(8,6))
	plt.loglog(f, psd**0.5, 'c')
        plt.loglog(f, abs(data),'r',lw=2, label='SXS')
        plt.loglog(f, abs(best_fit_signal), 'k',lw=2,alpha=0.7,label='phenomhm')
	plt.xlabel('$f$ [Hz]')
	plt.ylabel('$h(f)$ and $S_h(f)$')
	plt.xlim([f_low, 200.])
        plt.ylim([5e-25, 5e-23])
        plt.xlabel('f')
        plt.ylabel('$\\tilde{h}(f)$')
        plt.legend(loc='best')
        plt.title('BBH, four modes with inclination, $\iota=%.2f$'%incl_angle)
        plt.savefig(out_dir + '/%s_data.png'%out_file)
	plt.close()

	# saving data
<<<<<<< HEAD
	np.savetxt(out_dir + '/%s_data.dat'%out_file, np.c_[f, datar, datai ,psd], header='f real_data imag_data psd')
	np.savetxt(out_dir + '/%s_initial.dat'%out_file, np.c_[Mc, q, r, iota, t0, Psi_ref, ra, np.sin(dec), pol], header='Mc q r iota t0 Psi_ref ra sin_dec pol')
=======
	np.savetxt(out_dir + '/%s_data.dat'%out_file, np.c_[f_SI[range(N/2)],datar[range(N/2)],datai[range(N/2)],psd[range(N/2)]], header='f real_data imag_data psd')
	np.savetxt(out_dir + '/%s_initial.dat'%out_file, np.c_[Mc, q, r, iota, t0, Psi_ref, ra, np.sin(dec), pol], header='Mc q r iota t0 Psi_ref ra sin_dec pol', fmt=['%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f'])
>>>>>>> e867df712b1f99e394e0e97af4c518cd1e3e95ae

	end_time = time.time()
	print "... time taken: %.2f seconds"%(end_time-start_time)