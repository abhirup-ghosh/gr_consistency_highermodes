"""
Template for test of GR in 2 variable case Mc+dMc, q+dq
"""

from numpy import sqrt, sin, cos, pi,exp
import matplotlib
matplotlib.use('Agg')
import numpy as np
#import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/abhirup/Documents/Work/phenom_hh/src')
import phenomhh as phh
from lal import MSUN_SI, MTSUN_SI, PC_SI, PI, PC_SI, C_SI, GAMMA, MRSUN_SI
import matplotlib.pyplot as plt


def ringdown(Mc,q,l,m,Ncs,df):
        M=((1.+q)**1.2)*Mc/(q**0.6)
        m1=M/(q+1.)
        m2=M*q/(q+1.)
        mt=m1+m2
        eta=m1*m2/(M*M)

        loc='../../phenom_hh/data/PhenFits/2017-07-28_v6280:6286M_uniformweight_amplfitover_fmin_fring_lambdaRd_phasefitover_fring_phasefitorder_v12LPN_Ampfitorder_4andhalfPN/2017-07-28_v6280:6286M_uniformweight_fring_amp_asym'

        if l==2 and m==1 or l==2 and m==2 or l==3 and m==3 or l==4 and m==4 or l==3 and m==2:

                filename = 'PhenParamFitsVsEta_l%d_m%d_phase_p.dat' %(l, m)

                psi8_coef, psi8L_coef, psi8L2_coef, psi9_coef, psi9L_coef, psi10_coef, psi10L_coef, psi11_coef, psi11L_coef, psi12_coef, psi12L_coef, psi12L2_coef, f_ring_coef = np.loadtxt(loc+'/'+filename, unpack=True)

        else:
                raise ValueError('Unknown mode (l=%d, m=%d)' % (l, m))
        eta_p2=eta**2.
        eta_p3=eta_p2*eta

        f_ring =  f_ring_coef[2] + f_ring_coef[1]*eta + f_ring_coef[0]*eta_p2
        f_ring = f_ring/(mt*MTSUN_SI) #SI unit conversion

        N = np.int(f_ring/df)+Ncs
        return N






def phenomhh_waveform_SI(Mc,q,Mc1,q1,r,iota,t0,phase,f_low,df,Ncs):

	N = np.max(np.array([ringdown(Mc,q,2,2,Ncs,df),ringdown(Mc1,q1,2,1,Ncs,df),ringdown(Mc1,q1,3,3,Ncs,df),ringdown(Mc1,q1,4,4,Ncs,df)]))
        N = np.int(N)

	M=((1.+q)**1.2)*Mc/(q**0.6)
	m1=M/(q+1.)
	m2=M*q/(q+1.)
	m1=m1*MSUN_SI 
	m2=m2*MSUN_SI
	mt=m1+m2 # mtotal


        M1=((1.+q1)**1.2)*Mc1/(q1**0.6)
        m1_1=M1/(q1+1.)
        m2_2=M1*q1/(q1+1.)
        m1_1=m1_1*MSUN_SI 
        m2_2=m2_2*MSUN_SI
        mt_1=m1_1+m2_2 # mtotal

	incl_angle=iota # inclination angle
	phi=0. 
	lmax=4

	Psi_ref = phase #initial phase	

	hpf22,hcf22 = phh.generate_phenomhmv1_fd(m1, m2, incl_angle, phi, f_low, df, N, lmax,[[2,2]], Psi_ref) 
	hpf21,hcf21 = phh.generate_phenomhmv1_fd(m1_1, m2_2, incl_angle, phi, f_low, df, N, lmax,[[2,1]], Psi_ref)
	hpf33,hcf33 = phh.generate_phenomhmv1_fd(m1_1, m2_2, incl_angle, phi, f_low, df, N, lmax,[[3,3]], Psi_ref)
	hpf44,hcf44 = phh.generate_phenomhmv1_fd(m1_1, m2_2, incl_angle, phi, f_low, df, N, lmax,[[4,4]], Psi_ref)
	
	f=np.linspace(0., df*(N-1), N)
	
	hpf22=hpf22*mt*MRSUN_SI*MTSUN_SI*mt*exp(-2*pi*1j*f*t0)/(MSUN_SI*MSUN_SI*(1.0e6*r*PC_SI))
	hcf22=hcf22*mt*MRSUN_SI*MTSUN_SI*mt*exp(-2*pi*1j*f*t0)/(MSUN_SI*MSUN_SI*(1.0e6*r*PC_SI))

        hpf21=hpf21*mt_1*MRSUN_SI*MTSUN_SI*mt_1*exp(-2*pi*1j*f*t0)/(MSUN_SI*MSUN_SI*(1.0e6*r*PC_SI))
        hcf21=hcf21*mt_1*MRSUN_SI*MTSUN_SI*mt_1*exp(-2*pi*1j*f*t0)/(MSUN_SI*MSUN_SI*(1.0e6*r*PC_SI))

        hpf33=hpf33*mt_1*MRSUN_SI*MTSUN_SI*mt_1*exp(-2*pi*1j*f*t0)/(MSUN_SI*MSUN_SI*(1.0e6*r*PC_SI))
        hcf33=hcf33*mt_1*MRSUN_SI*MTSUN_SI*mt_1*exp(-2*pi*1j*f*t0)/(MSUN_SI*MSUN_SI*(1.0e6*r*PC_SI))

        hpf44=hpf44*mt_1*MRSUN_SI*MTSUN_SI*mt_1*exp(-2*pi*1j*f*t0)/(MSUN_SI*MSUN_SI*(1.0e6*r*PC_SI))
        hcf44=hcf44*mt_1*MRSUN_SI*MTSUN_SI*mt_1*exp(-2*pi*1j*f*t0)/(MSUN_SI*MSUN_SI*(1.0e6*r*PC_SI))


	hpf=hpf22+hpf21+hpf33+hpf44
	hcf=hcf22+hcf21+hcf33+hcf44


	return f,hpf,hcf

