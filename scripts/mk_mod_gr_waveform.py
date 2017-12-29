"""
program to generate phenomhh_waveform in SI units given masses and distances. Remember to source Ajithm profile before running the program
"""

from numpy import sqrt, sin, cos, pi,exp
import matplotlib
matplotlib.use('Agg')
import numpy as np
#import matplotlib.pyplot as plt
import phenomhh as phh
from lal import MSUN_SI, MTSUN_SI, PC_SI, PI, PC_SI, C_SI, GAMMA, MRSUN_SI
import pycbc.waveform as pw
import pycbc.filter.matchedfilter as mfilter
import pycbc.psd
import pycbc.noise.gaussian
from pycbc  import  detector
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pycbc.types.frequencyseries
from pycbc.types import TimeSeries, FrequencySeries, zeros

f_low=20. #lower cutoff frequency
df=0.1
N=10000

def phenomhh_waveform_SI(M,q,r,iota,f_low,df,N,t0):

        m1=M/(q+1.)
        m2=M*q/(q+1.)
        m1=m1*MSUN_SI #q=m1/m2 >1. Always m1 should be higher in the code
        m2=m2*MSUN_SI
        mt=m1+m2 # mtotal
        incl_angle= iota # inclination angle
        phi=0.
        lmax=4
        Psi_ref= pi
        f=np.linspace(0., df*(N-1), N)
        hpf_22,hcf_22 = phh.generate_phenomhmv1_fd(m1, m2, incl_angle, phi, f_low, df, N, lmax,[[2,2]], Psi_ref)
        h_r_22=(hpf_22+1j*hcf_22)/np.sqrt(2)
        h_l_22=(hpf_22-1j*hcf_22)/np.sqrt(2)

        hpf_33,hcf_33 = phh.generate_phenomhmv1_fd(m1, m2, incl_angle, phi, f_low, df, N, lmax,[[3,3]], Psi_ref)
        h_r_33=(hpf_33+1j*hcf_33)/np.sqrt(2)
        h_l_33=(hpf_33-1j*hcf_33)/np.sqrt(2)

        hpf_44,hcf_44 = phh.generate_phenomhmv1_fd(m1, m2, incl_angle, phi, f_low, df, N, lmax,[[4,4]], Psi_ref)
        h_r_44=(hpf_44+1j*hcf_44)/np.sqrt(2)
        h_l_44=(hpf_44-1j*hcf_44)/np.sqrt(2)

        hpf_21,hcf_21 = phh.generate_phenomhmv1_fd(m1, m2, incl_angle, phi, f_low, df, N, lmax,[[2,1]], Psi_ref)
        h_r_21=(hpf_21+1j*hcf_21)/np.sqrt(2)
        h_l_21=(hpf_21-1j*hcf_21)/np.sqrt(2)

        dphase = 0.001*f
#       h_r=h_r*exp(1j*pi/3)
#       h_l=h_l*exp(-1j*pi/3)
        h_r = h_r_22*exp(-2*dphase)+h_r_21*exp(-1*dphase)+h_r_33*exp(-3*dphase)+h_r_44*exp(-4*dphase)
        h_l = h_l_22*exp(2*dphase)+h_l_21*exp(1*dphase)+h_l_33*exp(3*dphase)+h_l_44*exp(4*dphase)
        hpf = (h_r+h_l)/np.sqrt(2)
        hcf = ((h_r-h_l)*(-1j))/np.sqrt(2)

        hpf=hpf*mt*MRSUN_SI*MTSUN_SI*mt*exp(-2*pi*1j*f*t0)/(MSUN_SI*MSUN_SI*(1.0e6*r*PC_SI))
        hcf=hcf*mt*MRSUN_SI*MTSUN_SI*mt*exp(-2*pi*1j*f*t0)/(MSUN_SI*MSUN_SI*(1.0e6*r*PC_SI))

	return f,hpf,hcf

M=80.
q=1./9
r=1.
iota=pi/2
f,hpf,hcf= phenomhh_waveform_SI(M,q,r,iota,f_low,df,N,6.)

ra=1.
dec =1.
pol=0.
Fp,Fc = detector.overhead_antenna_pattern(ra, dec, pol)
psd = pycbc.psd.aLIGOZeroDetHighPower(len(hpf), df, 20.)
signal=Fp*hpf+Fc*hcf
signal_freq=pycbc.types.frequencyseries.FrequencySeries(signal,delta_f=0.1,dtype=complex)#mfilter.make_frequency_series(signal)
SNR=mfilter.sigma(signal_freq,psd=psd,low_frequency_cutoff=f_low)
print SNR,r

r=SNR*r/25.
f,hpf,hcf= phenomhh_waveform_SI(M,q,r,iota,f_low,df,N,6.)

ra=1.
dec =1.
pol=0.
Fp,Fc = detector.overhead_antenna_pattern(ra, dec, pol)
psd = pycbc.psd.aLIGOZeroDetHighPower(len(hpf), df, 20.)
signal=Fp*hpf+Fc*hcf
signal_freq=pycbc.types.frequencyseries.FrequencySeries(signal,delta_f=0.1,dtype=complex)#mfilter.make_frequency_series(signal)
SNR=mfilter.sigma(signal_freq,psd=psd,low_frequency_cutoff=f_low)
Mc=(M*q**0.6)/((1.+q)**1.2)
print SNR,r,Mc
n2=1
#while n2 <= 10:
noise=pycbc.noise.gaussian.frequency_noise_from_psd(psd, seed=None)
data=signal#+noise

datar=np.real(data)
datai=np.imag(data)

loc='/home/siddharth.dhanpal/Work/projects/imrtestgr_hh/runs/kludge_injections_pv_biref/M_80/q_9/i_90/data'#'/home/siddharth.dhanpal/Work/projects/imrtestgr_hh/runs/201711_pe_deviations_noise_free_analysis/M_80/q_9/i_90/data'
np.savetxt('%s/detected_data_kludge_pv1.txt'%loc,np.c_[f,datar,datai,psd])
#np.savetxt('%s/detected_data_kludge_pv_%d.txt'%(loc,n2),np.c_[f,datar,datai,psd])
#n2+=1
print signal
print signal_freq
"""
snr_series=4.*(np.abs(signal[200:-1])**2)*df/psd[200:-1]
snr_formula=np.sum(snr_series)
print np.sqrt(snr_formula)"""
#print signal_freq
"""
noise=pycbc.noise.gaussian.frequency_noise_from_psd(psd, seed=None)
data=signal+noise

print data
datar=np.real(data)
datai=np.imag(data)
np.savetxt('data_35_35_pycbc.txt',np.c_[f,datar,datai,psd])"""

