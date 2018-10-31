"""
Code to generate GR waveform
"""

from numpy import sqrt, sin, cos, pi,exp
import matplotlib
matplotlib.use('Agg')
import numpy as np
#import matplotlib.pyplot as plt
import phenomhh as phh
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

M=70.          ###### Total Mass in M_SUN
q=1./6         ###### Mass ratio
SNR_req=25.    ###### Required SNR
iota=0.
Psi_ref=0.
t0=0.          ###### time of arrival

ra=0.          ##### Sky localisation
dec =0.
pol=0.

################################################

f,hpf,hcf= GR_waveform_SI(M, q, 1., iota, Psi_ref, f_low, df, N, t0)

Fp,Fc = detector.overhead_antenna_pattern(ra, dec, pol)
psd = pycbc.psd.aLIGOZeroDetHighPower(len(hpf), df, f_low)

signal=Fp*hpf+Fc*hcf
signal_freq=pycbc.types.frequencyseries.FrequencySeries(signal,delta_f=df,dtype=complex)
SNR=mfilter.sigma(signal_freq,psd=psd,low_frequency_cutoff=f_low)

print 'SNR at 1Mpc is... %f'%SNR 

r=SNR/SNR_req
f,hpf,hcf= GR_waveform_SI( M, q, r, iota, Psi_ref, f_low, df, N, t0)

signal=Fp*hpf+Fc*hcf
signal_freq=pycbc.types.frequencyseries.FrequencySeries(signal,delta_f=df,dtype=complex)
SNR=mfilter.sigma(signal_freq,psd=psd,low_frequency_cutoff=f_low)

Mc=(M*q**0.6)/((1.+q)**1.2)

print 'The SNR is... %f'%SNR
print 'dL for the output SNR is.. %f Mpc'%r
print 'Chirp mass is.. %f solar mass'%Mc

##### Output location and data file name #######

'''
Description of output data:

Output data contains [freq ,Real.part of signal ,Imag.part of signal ,PSD] in the same order
Data is non-zero starting from f_low. Last entry of data is (N-2)*df

'''
out_file = 'GR_M_%.2f_Mc_%.2f_q_%.2f_snr_25_i_%.2f_dL_%.2f_phiref_%.2f_t0_%.2f_ra_%.2f_dec_%.2f_psi_%.2f_flow_20Hz_fig1.dat'%(M,Mc,q,iota, r, Psi_ref, t0, ra, dec, pol)

### Generating noise from psd ###
#noise=pycbc.noise.gaussian.frequency_noise_from_psd(psd, seed=None)
data=signal#+noise ## comment noise to generate noise free data
#################################

datar=np.real(data)
datai=np.imag(data)

np.savetxt(out_file, np.c_[f[:-1],datar[:-1],datai[:-1],psd[:-1]], header='f real_data imag_data psd')
