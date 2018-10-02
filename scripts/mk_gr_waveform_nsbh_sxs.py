import matplotlib as mpl
mpl.use('Agg')
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
import time

# signal parameters
f_low=20.

m1, m2 = 60.0, 10.0
M=70.0          # in MSUN
q=1./6         
Mc=(M*q**0.6)/((1.+q)**1.2)
SNR_req=25.    
iota_list=[0.00,0.79,1.57]
Psi_ref=np.pi
t0=6.

ra=1.          
dec =1.
pol_list=[-3.14,-1.57,0.00]

cbc_list = ['NSBH', 'BBH']

data_dir = '/home/abhirup/Documents/Work/gr_consistency_highermodes/data/polarizations'
out_dir = '/home/abhirup/Documents/Work/gr_consistency_highermodes/injections/modGR_simulations'

for cbc in cbc_list:
  for iota in iota_list:
    for pol in pol_list:
	start_time = time.time()

	out_file = '%s_iota_%.3f_pol_%.3f_Mtot_70_flow15'%(cbc, iota, pol)

	print "... case:", cbc, iota, pol

	data_loc = glob.glob(data_dir + '/NRPolzns_%s_SpEC_q6.00_*_iota_%.2f_psi_%.2f.dat.gz'%(cbc, iota, pol))[0]
	t_geom, hp_geom, hc_geom = np.loadtxt(data_loc, unpack=True)
	print "... read data"

	r = 1. #in Mpc

	t_SI = t_geom * (M*lal.MTSUN_SI)
	hp_SI = hp_geom*(M*lal.MTSUN_SI)/(r*1e6*lal.PC_SI/lal.C_SI)
	hc_SI = hc_geom*(M*lal.MTSUN_SI)/(r*1e6*lal.PC_SI/lal.C_SI)

	h_SI = hp_SI + 1j*hc_SI
	phi_SI = np.unwrap(np.angle(h_SI))
	Foft_SI = np.gradient(phi_SI)/np.gradient(t_SI)/(2*np.pi)

	idx_flow15, = np.where(Foft_SI > 15.)

	t_SI_flow15, hp_SI_flow15, hc_SI_flow15 = t_SI[idx_flow15], hp_SI[idx_flow15], hc_SI[idx_flow15]

	t_SI_flow15_interp = np.arange(t_SI_flow15[0], t_SI_flow15[-1], 1/16384.)
	dt_SI_flow15_interp = np.mean(np.diff(t_SI_flow15_interp))
        hp_SI_flow15_interp_obj = scipy.interpolate.interp1d(t_SI_flow15, hp_SI_flow15, fill_value=0., bounds_error=False)
        hc_SI_flow15_interp_obj = scipy.interpolate.interp1d(t_SI_flow15, hc_SI_flow15, fill_value=0., bounds_error=False)
        hp_SI_flow15_interp = hp_SI_flow15_interp_obj(t_SI_flow15_interp)
        hc_SI_flow15_interp = hc_SI_flow15_interp_obj(t_SI_flow15_interp)

	plt.figure(figsize=(10,10))
        plt.subplot(311)
        plt.plot(t_SI_flow15, hp_SI_flow15, 'k', lw=0.2)
        plt.plot(t_SI_flow15_interp, hp_SI_flow15_interp, 'r--',alpha=0.2, lw=0.2)
        plt.subplot(312)
        plt.plot(t_SI_flow15, hc_SI_flow15, 'k', lw=0.2)
        plt.plot(t_SI_flow15_interp, hc_SI_flow15_interp, 'r--',alpha=0.2, lw=0.2)
	plt.subplot(313)
        plt.plot(t_SI, Foft_SI, 'k', lw=0.2)
	plt.savefig(out_dir + '/%s_interpolation.png'%out_file, dpi=300)
        plt.close()

	N = len(hp_SI_flow15_interp)
	f_SI = np.fft.fftfreq(N, d=dt_SI_flow15_interp)
	df = np.diff(f_SI)[0]
	Fp,Fc = detector.overhead_antenna_pattern(ra, dec, pol)
        psd = pycbc.psd.aLIGOZeroDetHighPower(N, df, f_low)

	signal=Fp*hp_SI_flow15_interp+Fc*hc_SI_flow15_interp
	signal_time=pycbc.types.timeseries.TimeSeries(signal,delta_t=dt_SI_flow15_interp,dtype=float)
        SNR=mfilter.sigma(signal_time,psd=psd,low_frequency_cutoff=f_low)

	print '... initial SNR:%f'%SNR

	r = SNR/SNR_req

        hp_SI_rescaled = hp_SI_flow15_interp/r#hp_geom_interp*(M*lal.MTSUN_SI)/(r*1e6*lal.PC_SI/lal.C_SI)
        hc_SI_rescaled = hc_SI_flow15_interp/r#hc_geom_interp*(M*lal.MTSUN_SI)/(r*1e6*lal.PC_SI/lal.C_SI)

	signal=Fp*hp_SI_rescaled+Fc*hc_SI_rescaled
        signal_time=pycbc.types.timeseries.TimeSeries(signal,delta_t=dt_SI_flow15_interp,dtype=float)
        SNR=mfilter.sigma(signal_time,psd=psd,low_frequency_cutoff=f_low)

	print '... rescaled distance: %f Mpc for a fixed SNR: %f'%(r, SNR)

	signal_freq = np.fft.fft(signal)*dt_SI_flow15_interp
	data=signal_freq#+noise ## comment noise to generate noise free data

	datar=np.real(data)
	datai=np.imag(data)

	plt.figure(figsize=(8,6))
	plt.loglog(f_SI, abs(data), 'r')
	plt.loglog(f_SI, psd**0.5, 'c')
	plt.xlabel('$f$ [Hz]')
	plt.ylabel('$h(f)$ and $S_h(f)$')
	plt.savefig(out_dir + '/%s_data.png'%out_file)
	plt.close()

	np.savetxt(out_dir + '/%s_data.dat'%out_file, np.c_[f_SI[range(N/2)],datar[range(N/2)],datai[range(N/2)],psd[range(N/2)]], header='f real_data imag_data psd')
	np.savetxt(out_dir + '/%s_initial.dat'%out_file, np.c_[Mc, q, Mc, q, r, iota, t0, Psi_ref, ra, np.sin(dec), pol], header='Mc q Mc1 q1 r iota t0 Psi_ref ra sin_dec pol')

	end_time = time.time()
	print "... time taken: %.2f seconds"%(end_time-start_time)
