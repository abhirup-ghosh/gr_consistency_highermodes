"""
Code to generate GR waveform
"""
import sys
from numpy import sqrt, sin, cos, pi,exp
import matplotlib
matplotlib.use('Agg')
import numpy as np
#import matplotlib.pyplot as plt
#import phenomhh_tgr as phh
# FROM ABHIRUP
#sys.path.append('/home/abhirup/Documents/Work/phenom_hh/src')
#import phenomhh as phh

#FROM AJIT
sys.path.append('/home/ajithm/Ajit_work/phenom_hh/src')
import phenomhh_tgr as phh

from lal import MSUN_SI, MTSUN_SI, PC_SI, PI, PC_SI, C_SI, GAMMA, MRSUN_SI
import pycbc.filter.matchedfilter as mfilter
import pycbc.psd
import pycbc.noise.gaussian
from pycbc  import  detector
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pycbc.types.frequencyseries
from pycbc.types import TimeSeries, FrequencySeries, zeros



out_dir = 'injections_bank_diff_iota'



def GR_waveform_SI(M,q,r,iota,Psi_ref,f_low,df,N,t0):
	"""	
	Inputs: Total mass  M
		Mass ratio  q
		inlination angle  iota
		lower freq  f_low
		frequency resol	 df
		Total_number of points N where (N-2)*df is  f_high
		time of arrival  t0
		initial phase  Psi_ref


	Outputs: f,hpf,hcf GR waveform

	"""

        m1=M/(q+1.)
        m2=M*q/(q+1.)
        m1=m1*MSUN_SI 
        m2=m2*MSUN_SI
        mt=m1+m2 
        incl_angle= iota 
        phi=0.
        lmax=4

        f=np.linspace(0., df*(N-1), N)
        hpf,hcf = phh.generate_phenomhmv1_fd(m1, m2, incl_angle, phi, f_low, df, N, lmax,[[2,2],[2,1],[3,3],[4,4]], Psi_ref)

        hpf=hpf*mt*MRSUN_SI*MTSUN_SI*mt*exp(-2*pi*1j*f*t0)/(MSUN_SI*MSUN_SI*(1.0e6*r*PC_SI))
        hcf=hcf*mt*MRSUN_SI*MTSUN_SI*mt*exp(-2*pi*1j*f*t0)/(MSUN_SI*MSUN_SI*(1.0e6*r*PC_SI))

	return f,hpf,hcf




###### MAIN ##########
### parameters of the output data####
f_low=20.
df=0.1
N=10000

Minj=40.          ###### Total Mass in M_SUN
Mlist=[40.,60.,80.,100.,120.,140.,160.,180.,200.]
qlist=[1./1.,2./3.,1./2.,1./3.,1./4.,1./5.,1./6.,1./7.,1./8.,1./9.]         ###### Mass ratio
qinj=1./9.0
SNR_req=25.    ###### Required SNR
iotalist=[pi/6.0,pi/4.0,pi/3.0,80.0*pi/180.,pi/2.0]
Psi_ref=pi

t0=6.0#1126285216.          ###### time of arrival at the earth center
t_gps=t0

ra=1.0          ##### Sky localisation
dec =1.0
pol=0.

################################################
###  generte waveform
def generate_waveform(M,q,iota):
	# three  detectors
	H = detector.Detector("H1")
	L = detector.Detector("L1")
	V = detector.Detector("V1")
	# compute antenna patterns 
	Fp1,Fc1 = H.antenna_pattern(ra, dec, pol, t_gps)
	Fp2,Fc2 = L.antenna_pattern(ra, dec, pol, t_gps)
	Fp3,Fc3 = V.antenna_pattern(ra, dec, pol, t_gps)
	# time of arrival/ time delay in detectors wrt earth center
	t01=H.time_delay_from_earth_center(ra, dec, t_gps)
	t02=L.time_delay_from_earth_center(ra, dec, t_gps)
	t03=V.time_delay_from_earth_center(ra, dec, t_gps)
	# signal in three detectors [time shifted]
	f1,hpf1,hcf1= GR_waveform_SI(M, q, 1.0, iota, Psi_ref, f_low, df, N, t0+t01)
	f2,hpf2,hcf2= GR_waveform_SI(M, q, 1.0, iota, Psi_ref, f_low, df, N, t0+t02)
	f3,hpf3,hcf3= GR_waveform_SI(M, q, 1.0, iota, Psi_ref, f_low, df, N, t0+t03)

	psd = pycbc.psd.aLIGOZeroDetHighPower(len(hpf1), df, f_low)
	psdv = pycbc.psd.analytical.AdVDesignSensitivityP1200087(len(hpf3), df, f_low)

	signal1=Fp1*hpf1+Fc1*hcf1
	signal2=Fp2*hpf2+Fc2*hcf2
	signal3=Fp3*hpf3+Fc3*hcf3

	signal_freq1=pycbc.types.frequencyseries.FrequencySeries(signal1,delta_f=df,dtype=complex)
	SNR1a=mfilter.sigma(signal_freq1,psd=psd,low_frequency_cutoff=f_low)

	signal_freq2=pycbc.types.frequencyseries.FrequencySeries(signal2,delta_f=df,dtype=complex)
	SNR2a=mfilter.sigma(signal_freq2,psd=psd,low_frequency_cutoff=f_low)

	signal_freq3=pycbc.types.frequencyseries.FrequencySeries(signal3,delta_f=df,dtype=complex)
	SNR3a=mfilter.sigma(signal_freq3,psd=psdv,low_frequency_cutoff=f_low)

	SNR_tota=np.sqrt(SNR1a*SNR1a+SNR2a*SNR2a+SNR3a*SNR3a)
	print 'H : SNR at 1Mpc is... %f'%SNR1a 
	print 'L : SNR at 1Mpc is... %f'%SNR2a
	print 'V : SNR at 1Mpc is... %f'%SNR3a
	print 'total SNR at 1Mpc is... %f'%SNR_tota

	####################################################
	# now scale it according to required snr

	r1=SNR_tota/SNR_req

	f1,hpf1,hcf1= GR_waveform_SI(M, q, r1, iota, Psi_ref, f_low, df, N, t0+t01)
	f2,hpf2,hcf2= GR_waveform_SI(M, q, r1, iota, Psi_ref, f_low, df, N, t0+t02)
	f3,hpf3,hcf3= GR_waveform_SI(M, q, r1, iota, Psi_ref, f_low, df, N, t0+t03)


	signal1=Fp1*hpf1+Fc1*hcf1
	signal2=Fp2*hpf2+Fc2*hcf2
	signal3=Fp3*hpf3+Fc3*hcf3

	signal_freq1=pycbc.types.frequencyseries.FrequencySeries(signal1,delta_f=df,dtype=complex)
	SNR1=mfilter.sigma(signal_freq1,psd=psd,low_frequency_cutoff=f_low)

	signal_freq2=pycbc.types.frequencyseries.FrequencySeries(signal2,delta_f=df,dtype=complex)
	SNR2=mfilter.sigma(signal_freq2,psd=psd,low_frequency_cutoff=f_low)

	signal_freq3=pycbc.types.frequencyseries.FrequencySeries(signal3,delta_f=df,dtype=complex)
	SNR3=mfilter.sigma(signal_freq3,psd=psdv,low_frequency_cutoff=f_low)

	SNR_tot=np.sqrt(SNR1*SNR1+SNR2*SNR2+SNR3*SNR3)

	Mc=(M*q**0.6)/((1.+q)**1.2)
	print 'H : SNR at 1Mpc is... %f'%SNR1a
	print 'L : SNR at 1Mpc is... %f'%SNR2a
	print 'V : SNR at 1Mpc is... %f'%SNR3a
	print 'total SNR at 1Mpc is... %f'%SNR_tota


	print 'dL for the output SNR is.. %f Mpc'%r1
	print 'The SNR in H1 is... %f'%SNR1
	print 'The SNR in L1 is... %f'%SNR2
	print 'The SNR in V1 is... %f'%SNR3
	print 'total SNR at distance dL is... %f'%SNR_tot

	print 'Chirp mass is.. %f solar mass'%Mc
	print 'time delay at H : %f'%t01
	print 'time delay at L : %f'%t02
	print 'time delay at V : %f'%t03

	##### Output location and data file name #######

	'''
	Description of output data:

	Output data contains [freq ,Real.part of signal ,Imag.part of signal ,PSD] in the same order
	Data is non-zero starting from f_low. Last entry of data is (N-2)*df

	'''
	out_file = 'GR_M_%.2f_q_%.2f_iota_%.2f_flow_20Hz'%(M,q,iota)
	### Generating noise from psd ###
	#noise=pycbc.noise.gaussian.frequency_noise_from_psd(psd, seed=None)
	data1=signal1#+noise ## comment noise to generate noise free data
	data2=signal2
	data3=signal3
	#################################

	plt.figure(figsize=(8,6))
	plt.loglog(f1, abs(signal1), 'g',linewidth=1.0,label='H1')
	plt.loglog(f2, abs(signal2), 'k',linewidth=1.0,label='L1')
	plt.loglog(f3, abs(signal3), 'r',linewidth=1.0,label='V1')
	plt.text(25,7e-25,'SNR in H1 : %f'%SNR1)
	plt.text(25,5e-25,'SNR in L1 : %f'%SNR2)
	plt.text(25,4e-25,'SNR in V1 : %f'%SNR3)
	plt.xlim(20,400)
	plt.ylim(5e-26,3e-22)
	plt.xlabel('$f$ [Hz]')
	plt.ylabel('signal')
	plt.title('$M=%.2f,q=%1.2f$'%(M,q))
	plt.legend()
	#plt.savefig(out_dir + '/M_%.2f_q_%.2f_iota_%.2f_signal_three_detectors.png'%(M,q,iota), dpi=200)
	
	datar1=np.real(data1)
	datai1=np.imag(data1)
	datar2=np.real(data2)
	datai2=np.imag(data2)
	datar3=np.real(data3)
	datai3=np.imag(data3)
	
#	np.savetxt(out_dir + '/%s_H1.dat'%out_file, np.c_[f1[:-1],datar1[:-1],datai1[:-1],psd[:-1]], header='f real_data imag_data psd')
#	np.savetxt(out_dir + '/%s_L1.dat'%out_file, np.c_[f2[:-1],datar2[:-1],datai2[:-1],psd[:-1]], header='f real_data imag_data psd')
#	np.savetxt(out_dir + '/%s_V1.dat'%out_file, np.c_[f3[:-1],datar3[:-1],datai3[:-1],psd[:-1]], header='f real_data imag_data psd')
#	np.savetxt(out_dir + '/%s_hm_initial.dat'%out_file, np.c_[Mc, q, Mc, q, r1, iota, t0,  Psi_ref, ra, np.sin(dec), pol], header='Mc, q, Mc, q, dL, iota, t0, Psi_ref, ra, dec, pol')
	return 0.0



##-------------------------------------------------------------------
# Injection bank
#--------------------------------------------------------------------

for q in qlist:
	for iota in iotalist:
		value=generate_waveform(Minj,q,iota)		

for M in Mlist:
        for iota in iotalist:
                value=generate_waveform(M,qinj,iota)



