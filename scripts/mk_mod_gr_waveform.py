"""
Code to generate mod_GR waveform
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


def modGR_waveform_SI(M,q,r,iota,Psi_ref,f_low,df,N,t0):
	"""
	Generates a modGR waveform for dphase=0.001*f 

	Inputs: Total mass M
		Mass ratio q
		inlination angle iota
		lower freq f_low
		frequency resol	df
		Total_number of points N where (N-2)*df is f_high
		time of arrival t0
		initial phase Psi_ref

	Generates h_R--> right circularly polarised
		  h_l--> left  circularly polarised 
	waveforms from phenomhh for each mode
	
	Modifies h_r and h_l where h_l is amplified and h_r is damped based on mode
	
	Outputs: f,hpf,hcf for mod GR waveform


	-->*To generate a GR waveform use dphase=0 inside
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

	#### Quantitative modification [dphase = 0 gives GR waveform]####
        dphase = 0.001*f

	######### amplitude modification of h_r and h_l##########
        mod_hpf = hpf + 1j*dphase*hcf
        mod_hcf = hcf - 1j*dphase*hpf
	
	######### SI units of h_plus and h_cross######
        mod_hpf=mod_hpf*mt*MRSUN_SI*MTSUN_SI*mt*exp(-2*pi*1j*f*t0)/(MSUN_SI*MSUN_SI*(1.0e6*r*PC_SI))
        mod_hcf=mod_hcf*mt*MRSUN_SI*MTSUN_SI*mt*exp(-2*pi*1j*f*t0)/(MSUN_SI*MSUN_SI*(1.0e6*r*PC_SI))

	return f,mod_hpf,mod_hcf




###### MAIN ##########
### parameters of the output data####
f_low=20.
df=0.1
N=20000

M=20.          ###### Total Mass in M_SUN
q=1./2         ###### Mass ratio
r=200.         ###### Luminosity distance in Mpc
iota=0.
Psi_ref=pi
t0=6.          ###### time of arrival

ra = 0.          ##### Sky localisation
dec = 0.
pol = pi/8


##### Output location and data file name #######

loc ='/home/siddharth.dhanpal/Work/projects/imrtestgr_hh/runs/kludge_injections_pv_biref/M_20/q_2/i_0/data'
data_fname = 'detected_data_mod_GR.txt'

'''
Description of output data:

Output data contains [freq ,Real.part of signal ,Imag.part of signal ,PSD] in the same order
Data is non-zero starting from f_low. Last entry of data is (N-2)*df

'''

################################################

Fp,Fc = detector.overhead_antenna_pattern(ra, dec, pol)
psd = pycbc.psd.aLIGOZeroDetHighPower(len(hpf), df, f_low)

f,hpf,hcf= modGR_waveform_SI( M, q, r, iota, Psi_ref, f_low, df, N, t0)

signal=Fp*hpf+Fc*hcf
signal_freq=pycbc.types.frequencyseries.FrequencySeries(signal,delta_f=df,dtype=complex)
SNR=mfilter.sigma(signal_freq,psd=psd,low_frequency_cutoff=f_low)

Mc=(M*q**0.6)/((1.+q)**1.2)

print 'The SNR is... %f'%SNR
print 'dL for the output SNR is.. %f Mpc'%r
print 'Chirp mass is.. %f solar mass'%Mc

### Generating noise from psd ###
#noise=pycbc.noise.gaussian.frequency_noise_from_psd(psd, seed=None)
data=signal#+noise ## comment noise to generate noise free data
#################################

datar=np.real(data)
datai=np.imag(data)

np.savetxt(loc+'/'+data_fname,np.c_[f[:-1],datar[:-1],datai[:-1],psd[:-1]])
