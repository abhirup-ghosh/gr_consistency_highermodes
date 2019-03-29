
"""FUNCTION TO GENERATE FOURIER-DOMAIN PLUS AND CROSS POLARIZATIONS OF THE GRAVITATIONAL WAVEFORMS"""
# function -  hpfhcf_phen(f, eta, t0, phi0, chi1, chi2, ampO, phaseO, iota, phi, qCase, lmax)
# f: fourier frequency
# eta: symmetric mass ratio param
# d_eff: effective distance to binary in Mpc
# chi1 and chi2: spin params of individual BHs
# t0: time at a refernce epoch 
# phi0: phase at a reference epoch 
# phaseO: PN accuracy of the phase
# ampO: PN accuracy of the amplitude 
# iota: inclination angle of the binary
# phi: initial phase angle of the binary
# qCase: index to mass-ratio cases under consideration
# lmax: highest "l" mode under consideration
import sys
sys.path.append('/home/ajithm/Ajit_work/phenom_hh/src')
import numpy as np 
import spinm2SphHarm
import os, socket
import qnmfreqs_berti as qnmfreq
from lal import MSUN_SI, MTSUN_SI, PC_SI, PI, PC_SI, C_SI, GAMMA
import hyb_hlmf as HYB
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time 
import h5py
import pnmodesfreqdom
import finalmassandspin_eobnrv2 as finalmassandspin
import compute_mixing_coef as cmode
import reset_lin_ini_phase as resetPsi

###################################################################################################################
# PHENOM PHASE MODEL
###################################################################################################################

def tf2_phase(f, l, m, eta, t0, phi0, chi1, chi2, phaseO, Psi_ref):
  
  ieta = 1.  # 0 for test-particle, 1 for comparable-mass
  delta = (1.-4.*ieta*eta)**(1./2.) # always choose +ve root
  chi_s = 0.5*(chi1+chi2) # symmetric combination of BH spin params
  chi_a = 0.5*(chi1-chi2) # anti-symmetric combination of BH spin param 
  # PN parameter "v" as a function of f 
  v = (2.*np.pi*f/m)**(1./3.)
  v0 = (2.*np.pi*f[0]/m)**(1./3.)
  # spin related coefficients 
  beta = (113./12. - 19.*ieta*eta/3.)*chi_s + (113.*delta/12.)*chi_a
  sigma = (ieta*eta*((721./48. - 247./48.)*(chi_s*chi_s - chi_a*chi_a)) + 
        (1.-2*ieta*eta)*((719./96. - 233./96.)*(chi_s*chi_s + chi_a*chi_a)) + 
        delta*((719./48. - 233./48.)*(chi_s*chi_a)))
  gamma = ((732985./2268. - 24260.*ieta*eta/81. - 340.*ieta*eta*eta/9.)*chi_s + 
        (732985./2268. + 140.*ieta*eta/9.)*delta*chi_a)

  # TaylorF2 phasing coefficients
  psiNewt = 3./(256.*eta)
  psi = np.zeros(8) # let's not use psi[0]
  psi[1] = 0.
  psi[2] = (3715./756. + 55.*ieta*eta/9.)
  psi[3] = (4.*beta - 16.*np.pi)
  psi[4] = (15293365./508032. + 27145.*ieta*eta/504. + 3085.*ieta*eta*eta/72. - 10.*sigma)
  psi[5] = (38645.*np.pi/756. - 65.*np.pi*ieta*eta/9 - gamma)
  psi[6] = (11583231236531./4694215680. - (640.*np.pi**2.)/3. - (6848.*GAMMA)/21. + 
        (-5162.983708047263 + (2255*np.pi**2)/12.)*ieta*eta + (76055*(ieta*eta)**2)/1728. - 
        (127825.*(ieta*eta)**3)/1296. + (2270.*np.pi/3.)*(chi_s + chi_a*delta - 
        156.*eta*chi_s/227.))
  psi[7] = ((77096675.*np.pi)/254016. + (378515.*np.pi*ieta*eta)/1512. - 
        (74045.*np.pi*(ieta*eta)**2)/756.)
  psi6L = -6848./21.
  
  # corrections to phase due to logarithmic terms in amplitude   
  PsiTailCorr = (6.*v**3)*(1-(eta/2.)*v**2)*np.log(v/v0)
  
  # TaylorF2 phasing 
  Psi = 1.
  for k in range(1, phaseO+1):
    # 2.5PN and 3PN terms have a log(v) term in them
    if k==5:
      Psi = Psi + psi[k]*(v**k)*(1.+3.*np.log(v))
    elif k==6:
      Psi = Psi + psi[k]*(v**k) + psi6L*np.log(4*v)*(v**k)
    else:
      Psi = Psi + psi[k]*(v**k)         

  Psi = psiNewt*Psi*(v**-5) 
    
  Psi = m*Psi 

  return -Psi

"""FIT to the difference of Hyb and TF2 phase"""
def dphase_im_fit(f, l, m, eta, pol):
        v = (2.*np.pi*f/m)**(1./3.)
       
        loc='/home/ajithm/Ajit_work/phenom_hh/data/PhenFits/2017-07-28_v6280:6286M_uniformweight_amplfitover_fmin_fring_lambdaRd_phasefitover_fring_phasefitorder_v12LPN_Ampfitorder_4andhalfPN/2017-07-28_v6280:6286M_uniformweight_fring_amp_asym'
        
        # load the data containing the phenomenological coefficients 
        if l==2 and m==1  or l==2 and m==2 or l==3 and m==3 or l==4 and m==4 :

                filename = 'PhenParamFitsVsEta_l%d_m%d_phase_%s.dat' %(l, m, pol)

                psi8_coef, psi8L_coef, psi8L2_coef, psi9_coef, psi9L_coef, psi10_coef, psi10L_coef, psi11_coef, psi11L_coef, psi12_coef, psi12L_coef, psi12L2_coef, f_ring_coef = np.loadtxt(loc+'/'+filename, unpack=True)

        else:
                raise ValueError('Unknown mode (l=%d, m=%d)' % (l, m))

        # compute the phasing coefficients at different orders 
        eta_p2=eta**2.
        eta_p3=eta_p2*eta
        
        psi8   =  psi8_coef[2] + psi8_coef[1]*eta  + psi8_coef[0]*eta_p2
        psi8L  =  psi8L_coef[2] + psi8L_coef[1]*eta + psi8L_coef[0]*eta_p2
        psi8L2 =  psi8L2_coef[2] + psi8L2_coef[1]*eta + psi8L2_coef[0]*eta_p2
        psi9   =  psi9_coef[2] + psi9_coef[1]*eta + psi9_coef[0]*eta_p2
        psi9L  =  psi9L_coef[2] + psi9L_coef[1]*eta + psi9L_coef[0]*eta_p2
        psi10  =  psi10_coef[2] + psi10_coef[1]*eta + psi10_coef[0]*eta_p2
        psi10L =  psi10L_coef[2] + psi10L_coef[1]*eta + psi10L_coef[0]*eta_p2
        psi11  =  psi11_coef[2] +psi11_coef[1]*eta +psi11_coef[0]*eta_p2
        psi11L =  psi11L_coef[2] + psi11L_coef[1]*eta + psi11L_coef[0]*eta_p2
        psi12  =  psi12_coef[2] +psi12_coef[1]*eta +psi12_coef[0]*eta_p2
        psi12L =  psi12L_coef[2] + psi12L_coef[1]*eta + psi12L_coef[0]*eta_p2
        psi12L2 =  psi12L2_coef[2] + psi12L2_coef[1]*eta + psi12L2_coef[0]*eta_p2
        f_ring =  f_ring_coef[2] + f_ring_coef[1]*eta + f_ring_coef[0]*eta_p2

        # efficiently compute powers of v 
        v5 = v**5
        v8 = v5*v*v*v
        v9 = v8*v
        v10 = v9*v
        v11 = v10*v
        v12 = v11*v
        log_v = np.log(v)

        # compute the phase correction 
        dPsiP = (psi8 + psi8L*log_v + psi8L2*log_v*log_v)*v8 \
                                + (psi9 + psi9L*log_v)*v9 \
                                + (psi10 + psi10L*log_v)*v10 \
                                + (psi11 + psi11L*log_v)*v11  + (psi12 + psi12L*log_v + psi12L2*log_v*log_v)*v12

        dPsiP *= 3*m/(256.*eta*v5)

        return dPsiP, f_ring

"""Gives the phase model for the ringdown"""
def ringdown_phase_ingradients(f, eta, qCase, l, m):
	phi0=0.
	mt=1.
	mf, af, AA, Omegaa = finalmassandspin.finalBHmassspin(mt,eta)
	n = 0  # number of overtones
	Psi_bf = np.zeros(len(f), dtype=np.complex128)

	Omega = (mt/mf)*qnmfreq.qnmfreqs_berti(af, l, m, n)
	omega0 = np.real(Omega)
	tau = -1/np.imag(Omega)
	omega = 2.*np.pi*f	
	sigma = 1./tau
	num = (sigma-1j*omega)*np.cos(phi0)-omega0*np.sin(phi0)
	den = (sigma-1j*omega)**2 + omega0**2
	bnf = num/den;
	Psi_bf = bnf
		
	Psi_rd = -np.unwrap(np.angle(Psi_bf))
	return Psi_rd 


"""phenom phase"""
def phenom_phase(f, l, m, eta, t0, phi0, chi1, chi2, phaseO, pol, qCase, Psi_ref):

  
  t0 = time.time()
  v = (2.*np.pi*f/m)**(1./3)
  Psi = tf2_phase(f, l, m, eta, t0, phi0, chi1, chi2, phaseO, Psi_ref)
  t1 = time.time()
  print '......... generated taylorf2 phase in %f secs' %(t1-t0)
  
  dPsiP, f_ring = dphase_im_fit(f, l, m, eta, pol)

  Psi_phen_im=Psi + dPsiP
   
  # ringdown idx
  rd_idx, = np.where(f>=f_ring)
  f_ring_idx = rd_idx[0]
  

  # derivative of the insp-merger phenom phase at f_ring
  df = np.mean(np.diff(f))
  dPsi_phen_im = np.gradient(Psi_phen_im)/df
  dPsi_phen_im_f_ring = dPsi_phen_im[f_ring_idx]

  # qnm dependent phase and derivatives at f_ring
  Psi_bf = ringdown_phase_ingradients(f, eta, qCase, l, m)

  dPsi_bf = np.gradient(Psi_bf)/df
  dPsi_bf_f_ring = dPsi_bf[f_ring_idx]

  #	t_ring, phi_ring estimate by matching the derivative of the phase (inspiral-merger and ringdown phases)
  t_ring = (dPsi_phen_im_f_ring - dPsi_bf_f_ring)/(2*np.pi)
  phi_ring = Psi_phen_im[f_ring_idx] - 2*np.pi*f[f_ring_idx]*t_ring - Psi_bf[f_ring_idx]

  # ringdown phase
  Psi_rd = 2*np.pi*f*t_ring + phi_ring + Psi_bf

  Psi_phen = Psi_phen_im
  Psi_phen[rd_idx] = Psi_rd[rd_idx]

  return Psi, dPsiP, Psi_phen, f_ring 


"""PN-PADE Amplitude of 22 mode """
def amp22_pade07(f, l, m, eta, ampO):
  ieta = 1
  delta = (1.-4.*ieta*eta)**(1./2.)
  v = (2.*np.pi*f/m)**(1./3.)
  #
  a = np.zeros(8)
  a[1] = 0
  a[2] = 0
  a[3] = 0
  a[4] = 0
  a[5] = 0
  a[6] = 0
  a[7] = 0
  b = np.zeros(8, dtype=np.complex)
  b[1] = 0.
  b[2] = (969 - 1804*ieta*eta)/672
  b[3] = 0
  b[4] = ((44213383 - 15529416*ieta*eta + 23208432*ieta*eta**2))/8128512
  b[5] = ((1536*1j)*ieta*eta + 85*np.pi - 340*eta*np.pi)/64
  b[6] = ((10025159216695 - 38121233538620*ieta*eta + 1647529054800*ieta*eta**2 -
      414838082880*ieta*eta**3 - (1224609103872*1j)*np.pi + 1283085619200*ieta*eta*np.pi**2))/300429803520
  b[7] = (((42237440*1j)*ieta*eta - (531572736*1j)*ieta*eta**2 + 3166405*np.pi -
      35896560*ieta*eta*np.pi + 119138160*ieta*eta**2*np.pi))/5806080
  amp_num = 1
  amp_den = 1
  for k in range(1, ampO+1):
    amp_num = amp_num + a[k]*v**k
    amp_den = amp_den + b[k]*v**k
  # Normalized amplitude of the 22 mode
  amp22 = abs(amp_num/amp_den)
  # Newtonian amplitude of the 22 mode
  amp22_newt = (1./2.)*np.sqrt(2./3.)*np.sqrt(eta)*np.pi*v**(-7./2.)
  # Pade Amplitude of the 22 mode
  amp22pade07 = amp22_newt*(amp22)
  return amp22_newt, amp22, amp22pade07

"""PN-PADE Amplitude of 21 mode """
def amp21_pade05(f, l, m, eta, ampO):
  ieta = 1  # 0 for test-particle, 1 for comparable-mass
  delta = (1.-4.*ieta*eta)**(1./2.)
  v = (2.*np.pi*f/m)**(1./3.)
  #
  a = np.zeros(6)
  a[1] = 0
  a[2] = 0
  a[3] = 0
  a[4] = 0
  a[5] = 0
  b = np.zeros(6, dtype=np.complex)
  b[1] = 0
  b[2] = ((-335 - 1404*ieta*eta))/672
  b[3] = (1j + 2*np.pi + (4*1j)*np.log(2))/2
  b[4] = ((2984407 + 40603032*ieta*eta + 13945968*ieta*eta**2))/8128512
  b[5] = (-5*(67*1j - (3012*1j)*ieta*eta - 223*np.pi + 2124*ieta*eta*np.pi + (268*1j)*np.log(2) +
      (1392*1j)*ieta*eta*np.log(2)))/1344
  amp_num = 1
  amp_den = 1
  for k in range(1, ampO-1):
    amp_num = amp_num + a[k]*v**k
    amp_den = amp_den + b[k]*v**k
  # Normalized amplitude of the 21 mode
  amp21 = amp_num/amp_den
  # Leading amplitude of the 21 mode
  amp21_lead = ((1./2.)*np.sqrt(2.*eta/3.)*np.pi*v**(-7./2.))*(1j*np.sqrt(2)*delta*v/3.)
  # Pade Amplitude of the 21 mode
  amp21_pade05 = abs(amp21_lead*(amp21))
  return amp21_lead, abs(amp21), amp21_pade05

"""PN-PADE Amplitude of 33 mode """
def amp33_pade05(f, l, m, eta, ampO):
  ieta = 1.  # 0 for test-particle, 1 for comparable-mass
  delta = (1.-4.*ieta*eta)**(1./2.)
  v = (2.*np.pi*f/m)**(1./3.)
  #
  a = np.zeros(6)
  a[1] = 0
  a[2] = 0
  a[3] = 0
  a[4] = 0
  a[5] = 0
  b = np.zeros(6, dtype=np.complex)
  b[1] = 0
  b[2] = ((1945 - 2268*ieta*eta))/672
  b[3] = ((21*1j - 5*np.pi + (30*1j)*np.log(2) - (30*1j)*np.log(3)))/5
  b[4] = ((4822859617 - 2808226008*ieta*eta + 2126120976*ieta*eta**2))/447068160
  b[5] = ((1323378*1j + (1852424*1j)*ieta*eta - 170505*np.pi - 156492*ieta*eta*np.pi +
      (3781080*1j)*np.log(2) - (4408992*1j)*ieta*eta*np.log(2) - (3781080*1j)*np.log(3) +
      (4408992*1j)*ieta*eta*np.log(3) + (1890540*1j)*(-np.log(2) + np.log(3)) -
      (1877904*1j)*ieta*eta*(-np.log(2) + np.log(3))))/108864
  amp_num = 1.
  amp_den = 1.
  for k in range(1, ampO-1):
    amp_num = amp_num + a[k]*v**k
    amp_den = amp_den + b[k]*v**k
  # Normalized amplitude of the 33 mode
  amp33 = amp_num/amp_den
  # Leading amplitude of the 33 mode
  amp33_lead = ((1./2.)*np.sqrt(2.*ieta*eta/3.)*np.pi*v**(-7./2.))*(1j*(-3./4.)*np.sqrt(5./7.)*delta*v)
  # Pade Amplitude of the 33 mode
  amp33_pade05 = abs(amp33_lead*(amp33))
  return amp33_lead, abs(amp33), amp33_pade05

"""PN-PADE Amplitude of 44 mode """
def amp44_pade04(f, l, m, eta, ampO):
  ieta = 1.  # 0 for test-particle, 1 for comparable-mass
  delta = (1.-4.*ieta*eta)**(1./2.)
  v = (2.*np.pi*f/m)**(1./3.)
  #
  a = np.zeros(5)
  a[1] = 0
  a[2] = 0
  a[3] = 0
  a[4] = 0
  b = np.zeros(5, dtype=np.complex)
  b[1] = 0
  b[2] = ((158383 - 641105*ieta*eta + 446460*ieta*eta**2))/(36960.*(1. - 3.*ieta*eta))
  b[3] = ((336*1j - (1193*1j)*ieta*eta - 80*np.pi + 240*ieta*eta*np.pi - (320*1j)*np.log(2) +
      (960*1j)*ieta*eta*np.log(2)))/(40*(1 - 3*ieta*eta))
  b[4] = ((5783159561419 - 39063917867658*ieta*eta + 79692564529827*ieta*eta**2 -
      50133238734600*ieta*eta**3 + 20074904483760*ieta*eta**4))/(319653734400*(1 - 3*ieta*eta)**2)
  amp_num = 1.
  amp_den = 1.
  for k in range(1, ampO-2):
    amp_num = amp_num + a[k]*v**k
    amp_den = amp_den + b[k]*v**k
  # Normalized amplitude of the 33 mode
  amp44 = amp_num/amp_den
  # Leading amplitude of the 33 mode
  amp44_lead =  ((1./2.)*np.sqrt(2.*ieta*eta/3.)*np.pi*v**(-7./2.))*((-4./9.)*np.sqrt(10./7.)*(1-3*ieta*eta)*v**2)
  # Pade Amplitude of the 44 mode
  amp44_pade04 = abs(amp44_lead*(amp44))
  return amp44_lead, abs(amp44), amp44_pade04

def amp31_pade05(f, l, m, eta, ampO):
        ieta = 1.  # 0 for test-particle, 1 for comparable-mass
        delta = (1.-4.*ieta*eta)**(1./2.)
        v = (2.*np.pi*f/m)**(1./3.)
        a = np.zeros(6)
        b = np.zeros(6, dtype=np.complex)
        a[0] = 1.
        a[1] = 0.
        a[2] = 0.
        a[3] = 0.
        a[4] = 0.
        a[5] = 0.

        b[0] = 1.
        b[1] = 0.
        b[2] = (1049. - 476.*eta)/672.
        b[3] = (7*1j + 5*np.pi + (10.*1j)*np.log(2.))/5.
        b[4] = (127467437. + 214001928.*eta + 131333328.*eta**2)/89413632.
        b[5] =  ((14686.*1j + (56*1j)*eta + 19415.*np.pi - 43820.*eta*np.pi + (20980.*1j)*np.log(2.) - (16240.*1j)*eta*np.log(2.)))/6720.

        amp_num=0.
        amp_den=0.
        for k in range(0, ampO-1):
                amp_num = amp_num + a[k]*v**k
                amp_den = amp_den + b[k]*v**k

        amp31 = amp_num/amp_den

        amp31_lead = ((1/2.)*np.sqrt(2.*eta/3.)*np.pi*v**(-7/2.))*(1j*(1/12.)*np.sqrt(1/7.)*delta*v)

        amp31_pade05 = abs(amp31_lead*(amp31))


        return amp31_lead, abs(amp31), amp31_pade05

def amp32_pade04(f, l, m, eta, ampO):
	ieta = 1.  # 0 for test-particle, 1 for comparable-mass
	delta = (1.-4.*ieta*eta)**(1./2.)
	v = (2.*np.pi*f/m)**(1./3.)
        a = np.zeros(5)
        b = np.zeros(5, dtype=np.complex)
	a[0] = 1.
	a[1] = 0.
	a[2] = 0.
	a[3] = 0.
	a[4] = 0.

	b[0] = 1.
	b[1] = 0.
	b[2] = ((-10471. + 61625.*eta - 82460.*eta**2))/(10080.*(-1. + 3.*eta))
	b[3] = (((3.*1j)/5.)*(-5. + 22.*eta))/(-1. + 3.*eta)
	b[4] = ((6532988997. - 40885070230.*eta + 55919840045.*eta**2 + 3978735880.*eta**3 + 60980652880.*eta**4))/(2235340800.*(-1. + 3.*eta)**2)

	amp_num=0.
	amp_den=0.
	for k in range(0, ampO-2):
        	amp_num = amp_num + a[k]*v**k
    		amp_den = amp_den + b[k]*v**k

	amp32 = amp_num/amp_den
	
	amp32_lead = ((1/2.)*np.sqrt(2.*eta/3.)*np.pi*v**(-7./2.))*((1./3.)*np.sqrt(5./7.)*(1.-3.*eta)*v**2)

        amp32_pade04 = abs(amp32_lead*(amp32))


   	return amp32_lead, abs(amp32), amp32_pade04

def amp43_pade03(f, l, m, eta, ampO):
        ieta = 1.  # 0 for test-particle, 1 for comparable-mass
        delta = (1.-4.*ieta*eta)**(1./2.)
        v = (2.*np.pi*f/m)**(1./3.)

        a = np.zeros(4)
        b = np.zeros(4, dtype=np.complex)

        a[0] = 1.
        a[1] = 0.
        a[2] = 0.
        a[3] = 0.

        b[0] = 1.
        b[1] = 0.
        b[2] = (-18035. + 64770.*eta - 49672.*eta**2)/(7392.*(-1. + 2.*eta))
        b[3] = (-5184.*1j + (16301.*1j)*eta + 810.*np.pi - 1620.*eta*np.pi + (4860.*1j)*(-np.log(2) + np.log(3)) - (9720.*1j)*eta*(-np.log(2) + np.log(3)))/(810.*(-1. + 2.*eta))

        amp_num=0.
        amp_den=0.
        for k in range(0, ampO-3):
            amp_num = amp_num + a[k]*v**k
            amp_den = amp_den + b[k]*v**k

        amp43 = amp_num/amp_den

        amp43_lead = ((1./2)*np.sqrt(2*eta/3.)*np.pi*v**(-7./2))*(1j*(-3./4)*np.sqrt(3./35)*(1.-2.*eta)*delta*v**3)

        amp43_pade03 = abs(amp43_lead*(amp43))


        return amp43_lead, abs(amp43), amp43_pade03

"""FIT to the difference of Hyb and TF2 phase"""
def ramp_fit(f, l, m, eta, ampO):
        v = (2.*np.pi*f/m)**(1./3.)
        loc='/home/ajithm/Ajit_work/phenom_hh/data/PhenFits/2017-07-28_v6280:6286M_uniformweight_amplfitover_fmin_fring_lambdaRd_phasefitover_fring_phasefitorder_v12LPN_Ampfitorder_4andhalfPN/2017-07-28_v6280:6286M_uniformweight_fring_amp_asym'

        if l==2 and m==1 or l==2 and m==2 or l==3 and m==3 or l==4 and m==4:
                filename = 'PhenParamFitsVsEta_l%d_m%d_amp.dat' %(l, m)

                amp8_coef, amp8L_coef, amp9_coef, amp9L_coef, amp10_coef, amp10L_coef, f_ring_coef, f_cut_coef, lambdaRd_coef = np.loadtxt(loc+'/'+filename, unpack=True)

        else:
                raise ValueError('Unknown mode (l=%d, m=%d)' % (l, m))
        # compute the phasing coefficients at different orders 
        eta_p2 = eta**2
        eta_p3 = eta_p2*eta

        amp8   =  amp8_coef[2] + amp8_coef[1]*eta + amp8_coef[0]*eta_p2
        amp8L  =  amp8L_coef[2] + amp8L_coef[1]*eta + amp8L_coef[0]*eta_p2
        amp9   =  amp9_coef[2] + amp9_coef[1]*eta + amp9_coef[0]*eta_p2
        amp9L  =  amp9L_coef[2] + amp9L_coef[1]*eta + amp9L_coef[0]*eta_p2
        amp10  =  amp10_coef[2] + amp10_coef[1]*eta + amp10_coef[0]*eta_p2
        amp10L  =  amp10L_coef[2] + amp10L_coef[1]*eta + amp10L_coef[0]*eta_p2

        f_ring = f_ring_coef[2] + f_ring_coef[1]*eta + f_ring_coef[0]*eta_p2
        f_cut =  f_cut_coef[2] + f_cut_coef[1]*eta + f_cut_coef[0]*eta_p2
        lambdaRd = lambdaRd_coef[2] + lambdaRd_coef[1]*eta + lambdaRd_coef[0]*eta_p2

        v8 = v**8
        v9 = v8*v
        v10 = v9*v
        v11 = v10*v
        log_v = np.log(v)

        rAmp_fit = (amp8 + amp8L*log_v)*v8 \
                                + (amp9 + amp9L*log_v)*v9  + (amp10 + amp10L*log_v)*v10

        return rAmp_fit, f_ring, f_cut, lambdaRd



"""Gives the phenom model for the post-merger(ringdown) signal"""
def ampRD_fit(f, l, m, eta, ampO, qCase, lambdaRd):
	# final BH mass and spin
	#total mass
	mt = 1.
	mf, af, AA, Omegaa = finalmassandspin.finalBHmassspin(mt,eta)
	Omega =(mt/mf)*qnmfreq.qnmfreqs_berti(af, l, m, 0)
	omega = 2*np.pi*f
	omega0 = np.real(Omega)
	sigma = -np.imag(Omega)
	num = sigma*(sigma**2 + omega**2 + omega0**2)
	den1 = sigma**2 + (omega-omega0)**2	
	den2 = sigma**2 + (omega+omega0)**2
	Af_rd = np.exp(-lambdaRd*f)*num/(den1*den2)
	return Af_rd, omega0

def ampRD_fit_asym(f, l, m, eta, ampO, qCase, lambdaRd):
	# final BH mass and spin
	#total mass
	mt = 1.
	mf, af, AA, Omegaa = finalmassandspin.finalBHmassspin(mt,eta)
	Omega =(mt/mf)*qnmfreq.qnmfreqs_berti(af, l, m, 0)
	omega = 2*np.pi*f
	omega0 = np.real(Omega)
	sigma = -np.imag(Omega)
	num = np.sqrt(omega**2 + sigma**2)
	den = np.sqrt(4*sigma**2*omega**2 + (sigma**2 - omega**2 + omega0**2)**2)	
	Af_rd = np.exp(-lambdaRd*f)*num/den
	return Af_rd, omega0

"""Gives the phenom model for amplitude"""
def amp_phenom(f, l, m, eta, ampO, qCase):
	if l==2 and m==1:
		amp_insp=amp21_pade05(f, l, m, eta, ampO)
	elif l==2 and m==2:
		amp_insp=amp22_pade07(f, l, m, eta, ampO)
	elif l==3 and m==3:
		amp_insp=amp33_pade05(f, l, m, eta, ampO)
	elif l==4 and m==4:
		amp_insp=amp44_pade04(f, l, m, eta, ampO)
	elif l==3 and m==2:
		amp_insp=amp32_pade04(f, l, m, eta, ampO)
        elif l==4 and m==3:
                amp_insp=amp43_pade03(f, l, m, eta, ampO)
	else:
		print 'Unknown mode (l=%d, m=%d)' % (l, m)
	rAmp_fit, f_ring, f_cut, lambdaRd = ramp_fit(f, l, m, eta, ampO)
        print '..printing f_ring = %.2f'%f_ring
	Amp_rd, omega0 = ampRD_fit_asym(f, l, m, eta, ampO, qCase, lambdaRd)
	# phenom amplitude model
	Amp_phen = amp_insp[2]*(1+rAmp_fit)
	# ringdown vector indices
	rd_idx, = np.where(f>=f_ring)	
	f_ring_idx = rd_idx[0]
	Amp_rd = (Amp_rd/Amp_rd[f_ring_idx])*(Amp_phen[f_ring_idx])
	Amp_phen[rd_idx] = Amp_rd[rd_idx]
	return Amp_phen, amp_insp[2], Amp_rd, omega0, f_ring

# function appearing in SphericalHarmonic basis function of Spin weight -2 independent of azimuthal angle 
def dm2_lm(l, m, iota):
  dm2lm = spinm2SphHarm.spinm2_sphharm(int(l), int(m), iota, 0)
  return dm2lm

""" Various pieces of freq domain real and imaginary part of the signal """ 
def hpfhcf_phen_pieces(f, c2, c3, c4, l, m, eta, t0, phi0, chi1, chi2, ampO, phaseO, iota, phi, qCase, Psi_ref):
  
  dm2lmP = dm2_lm(l, m, iota)
  dm2lmM = dm2_lm(l, -m, iota)

  # phenom model for amplitude
  t0 = time.time()
 
  Af_phen, Af_insp, Af_rd, omega0, f_ring  =  amp_phenom(f, l, m, eta, ampO, qCase)

  Psi_insp, dphasefit_p, Psipf_phen, f_ring_phase = phenom_phase(f, l, m, eta, t0, phi0, chi1, chi2, phaseO, 'p', qCase, Psi_ref) 
 
  # phenom model for phase 
  t1 = time.time()
  print 'f0 = %f ini_offset_phen_p = %f' %(f[0], Psipf_phen[0])

  print '...... generated the phen ampl and phase in %f sec' %(t1-t0)

  hlmfp_phen = Af_phen * np.exp(1j*Psipf_phen)

  print '..printing Psi_ref'
  print Psi_ref
  
  # pieces of phenom model for h+ and hx to be summed over various l and m values
  if l==2 and m==2:
  	hpf = (dm2lmM*(-1)**l + dm2lmP)*hlmfp_phen*np.exp(-1j*m*Psi_ref)
  	hcf = (-1j)*(dm2lmM*(-1)**l - dm2lmP)*hlmfp_phen*np.exp(-1j*m*Psi_ref) 
  if l==2 and m==1:
	hpf = ((np.conj(c2))*dm2lmM*(-1)**l + (c2)*dm2lmP)*hlmfp_phen*np.exp(-1j*m*Psi_ref)
        hcf = (-1j)*((np.conj(c2))*dm2lmM*(-1)**l - (c2)*dm2lmP)*hlmfp_phen*np.exp(-1j*m*Psi_ref)
  if l==3 and m==3:
	hpf = ((np.conj(c3))*dm2lmM*(-1)**l + (c3)*dm2lmP)*hlmfp_phen*np.exp(-1j*m*Psi_ref)
        hcf = (-1j)*((np.conj(c3))*dm2lmM*(-1)**l - (c3)*dm2lmP)*hlmfp_phen*np.exp(-1j*m*Psi_ref)
  elif l==4 and m==4:
	hpf = ((np.conj(c4))*dm2lmM*(-1)**l + (c4)*dm2lmP)*hlmfp_phen*np.exp(-1j*m*Psi_ref)
        hcf = (-1j)*((np.conj(c4))*dm2lmM*(-1)**l - (c4)*dm2lmP)*hlmfp_phen*np.exp(-1j*m*Psi_ref)
  return hpf, hcf

"""freq domain real and imaginary part of the signal"""
def hpfhcf_phen(f, c2, c3, c4, eta, t0, phi0, chi1, chi2, ampO, phaseO, iota, phi, qCase, lmax, use_only_these_modes, df, Psi_ref):

  # initialize the numpy arrays 
     hpf = np.zeros(len(f), dtype=np.complex128)
     hcf = np.zeros(len(f), dtype=np.complex128)
   
     l_vec = [2,2,3,3,3,4,4,4,4]
     m_vec = [1,2,1,2,3,1,2,3,4]

     for mode_idx in range(len(l_vec)):							

	# get the l,m and hlm of this mode, compute Ylm 
	l,m = int(l_vec[mode_idx]),int(m_vec[mode_idx])
       		
				
	# restrict to certain l modes 
	if l <=lmax and ((use_only_these_modes==None) or ([l,m] in use_only_these_modes)): 
            	print '......... using template mode l = %d m  =  %d' %(l, m)
                # if negative m modes are not given, assume the system is 
                # non precessing and use conjugate of positive m mode

                hpf_pieces, hcf_pieces=hpfhcf_phen_pieces(f, c2, c3, c4, l, m, eta, t0, phi0, chi1, chi2, ampO, phaseO, iota, phi, qCase, Psi_ref)
                hpf = hpf + hpf_pieces
		hcf = hcf + hcf_pieces
                

            
           
     return hpf, hcf
      
           
""" Top level function to generate h_+(f) and h_x(f) of the IMRPhenomHMv1 model.  """
def generate_phenomhmv1_fd(m1, m2, c2, c3, c4, incl_angle, phi, f_low, df, N, lmax, use_only_these_modes, Psi_ref): 
  #c=0.
  # create a freq vector -- this has to start from 0 Hz and end at f_Nyquist 
  f = np.linspace(0., df*(N-1), N) #when calling in calcmatch.py
  print 'f_Nyq = %f' %(df*N)

  m1 = m1/MSUN_SI     # convert kgs to solar masses 
  m2 = m2/MSUN_SI
  mt = m1+m2 
  q =round(m1/m2, 4) 
  eta = q/(1.+q)**2.
  t0 = 0.
  chi1, chi2 = 0., 0.
  ampO, phaseO = 7, 7
  phi0 = 0. 
  phi=0.

  qCase_dic = {1.:0,2.:1,3.:2,4.:3,5.:4,6.:5,7.:6,8.:7,9.:8,10.:9}
  #qCase =qCase_dic[round(q)]
  qCase=0
   
  # plus and cross polarization waveforms in F-domain
  hpf = np.zeros_like(f, dtype=np.complex128)
  hcf = np.zeros_like(f, dtype=np.complex128)
 
  band_idx = f >= (f_low) 
  f=f[band_idx]
  print 'fM_band: %f Hz to %f Hz' %(f[0], f[-1])
 
  f = f*mt*MTSUN_SI 
  # use phenom model as template
  print 'fM_band: %f Hz to %f Hz' %(f[0], f[-1])
  
  t1 = time.time()
  hpf_phen, hcf_phen = hpfhcf_phen(f, c2, c3, c4, eta, t0, phi0, chi1, chi2, ampO, phaseO, incl_angle, phi, qCase, lmax, use_only_these_modes, df, Psi_ref)
  t2 = time.time()
  print '..... time taken to generate phenomhm = %f secs' %(t2-t1)
  hpf[band_idx] = hpf_phen
  hcf[band_idx] = hcf_phen 

  print 'mtot = %f eta = %f t0 = %f phi0 = %f chi1 = %f chi2 = %f ampO = %d phaseO = %d incl_angle = %f phi = %f q=%d lmax = %d' %(mt, eta, t0, phi0, chi1, chi2, ampO, phaseO, incl_angle, phi, round(q), lmax) 
  
  return hpf, hcf

