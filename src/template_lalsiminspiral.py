import matplotlib as mpl
mpl.use('Agg')
import os, lal, subprocess, scipy, numpy as np
from scipy.interpolate import interp1d
import standard_gwtransf as gw
import pycbc
from pycbc import waveform
import os

def lalsiminspiral_waveform_SI(mc,q,dL,iota,t0,phiref,flow,df,Ncs):

  try:
	m1, m2 = gw.comp_from_mcq(mc, q)
	hpf, hcf = waveform.get_fd_waveform(approximant="IMRPhenomPv2",mass1=m1, mass2=m2, distance=dL, inclination=iota, coa_phase=phiref, delta_f=df, f_lower=flow, f_final=df*Ncs)

	print '... generated waveform for m1=%.2f and m2=%.2f'%(m1,m2)

	f = hpf.sample_frequencies
	hpf = hpf.data*np.exp(-2*np.pi*1j*f*t0)
	hcf = hcf.data*np.exp(-2*np.pi*1j*f*t0)

	return f, hpf, hcf
  except RuntimeError:
	print '... not generated waveform for m1=%.2f and m2=%.2f'%(m1,m2)
  except TypeError:
	print '... not generated waveform for m1=%.2f and m2=%.2f'%(m1,m2)
