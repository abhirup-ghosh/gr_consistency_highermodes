"""
Plots the phenom_hh waveform modified by amplitude birefringence

P. Ajith, 2018-01-29
"""

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import plotsettings 
import numpy as np
import phenomhh as phh
import lal 
import lalinspiral.sbank.psds as psds
import standard_cosmology as sc

# input paramseters 
f_low = 20.
df = 0.1
N = 20000

M = 80.          			# Total Mass in M_SUN
q_vec = [1., 9.]			# vector of mass ratios 
dL = 400.         		# Luminosity distance in Mpc
phi0 = 0.							# initial phase of the binary 
incl_angle = np.pi/3.	# inclination angle 
theta = 0. 						# sky location - polar (detector frame) 
phi=0.								# sky location - azimuth (detector frame) 
psi = np.pi/6. 				# polarization angle 
lmax=4								# max l to be included in phenom_hh 

K_SC_vec = [0., 2e6/lal.C_SI]	# values of K_SC := (theta_0_dot - theta_0_ddot/H0)

# plotting specs 
col_vec = ['r', 'k']
lw_vec = [3, 1]
ls_vec = ['-', '-']
alpha_vec = [0.4, 1]

# antenna patters 
Fp = 0.5*(1 + np.cos(theta)**2)*np.cos(2*phi)*np.cos(2*psi) - np.cos(theta)*np.sin(2*phi)*np.sin(2*psi)
Fc = 0.5*(1 + np.cos(theta)**2)*np.cos(2*phi)*np.sin(2*psi) + np.cos(theta)*np.sin(2*phi)*np.cos(2*psi)
print '... Antenna patterns: Fp = %f Fc = %f' %(Fp, Fc)

# cosmological redshift corresponding to dL 
z = sc.redshift(dL*sc.H0/sc.c)

plt.figure(figsize=(4.5,4.25))

# loop over mass ratios 
for iQ, q in enumerate(q_vec):

	m1 = M/(q+1.)
	m2 = M*q/(q+1.)

	# generate the GR waveform 
	f = np.linspace(0., df*(N-1), N)
	hpf, hcf = phh.generate_phenomhmv1_fd(m1*lal.MSUN_SI, m2*lal.MSUN_SI, incl_angle, phi, f_low, df, N, lmax,[[2,2],[2,1],[3,3],[4,4]], phi0)

	# loop over different values of the dephasing (CS modification to GR) 
	for i, K_SC in enumerate(K_SC_vec): 

		# amplitude modification of h_r and h_l
		deltaphi = 1j*f*np.pi*z*K_SC			# dephasing due to ampl birefringence dphi := i f pi z (theta_0_dot - theta_0_ddot/H0) 
		mod_hpf = hpf + hcf*deltaphi
		mod_hcf = hcf - hpf*deltaphi 

		# compute the detector response 
		hf = mod_hpf*Fp + mod_hcf*Fc

		# scale to SI units 
		hf = hf*(M*lal.MTSUN_SI)**2./(dL*1e6*lal.PC_SI/lal.C_SI)

		# compute SNR 
		Sh = psds.noise_models['aLIGOZeroDetHighPower'](f)
		band_idx = f >= f_low 
		rho = 2*np.sqrt(df*np.sum(abs(hf[band_idx])**2/Sh[band_idx]))
		print '... SNR = ', rho 

		# plot 
		plt.loglog(f, abs(hf), color=col_vec[iQ], lw=lw_vec[i], alpha=alpha_vec[i], ls=ls_vec[i])
		plt.ylabel('$|h(f)|$')
		plt.xlim(20, 600)
		plt.ylim(1e-25,1e-22)

# save plot 
plt.text(220, 0.8e-23, '$ q = 1$')
plt.text(110, 1e-24, '$ q = 9$')
plt.xlabel('$f$ [Hz]')
plt.tight_layout()
plt.savefig('mod_GR_waveform.pdf')
plt.close()
