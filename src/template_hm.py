"""
template for only (2,2) mode
"""

from numpy import sqrt, sin, cos, pi,exp
import sys
sys.path.append('/home/ajit.mehta/Ajit_work/phenom_hh/src')
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


def ringdown(Mc,q,l,m,Ncs,df):
        M=((1.+q)**1.2)*Mc/(q**0.6)
        m1=M/(q+1.)
        m2=M*q/(q+1.)
        mt=m1+m2
        eta=m1*m2/(M*M)

        loc = '/home/ajit.mehta/gr_consistency_highermodes/phenom_data'

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




def phenomhh_waveform_SI(Mc,q,r,iota,t0,phase,f_low,df,Ncs):

	N=ringdown(Mc,q,2,2,Ncs,df)

	M=((1.+q)**1.2)*Mc/(q**0.6)
	m1=M/(q+1.)
	m2=M*q/(q+1.)
	m1=m1*MSUN_SI 
	m2=m2*MSUN_SI
	mt=m1+m2 
	incl_angle=iota 
	phi=0. 
	lmax=4
	Psi_ref=1.3
	
	hpf22,hcf22 = phh.generate_phenomhmv1_fd(m1, m2, incl_angle, phi, f_low, df, N, lmax,[[2,2],[2,1],[3,3],[4,4]], Psi_ref) 
	
	f=np.linspace(0., df*(N-1), N)
	
	hpf22=hpf22*mt*MRSUN_SI*MTSUN_SI*mt*exp(-2*pi*1j*f*t0)/(MSUN_SI*MSUN_SI*(1.0e6*r*PC_SI))
	hcf22=hcf22*mt*MRSUN_SI*MTSUN_SI*mt*exp(-2*pi*1j*f*t0)/(MSUN_SI*MSUN_SI*(1.0e6*r*PC_SI))

	hpf=hpf22
	hcf=hcf22

	return f,hpf,hcf

