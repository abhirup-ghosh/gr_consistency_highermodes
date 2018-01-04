"""
This file is to plot GR amplitude and mod_GR amplitude for the same configuration M=80 q=1/9 i=90 at fixed distance of 200Mpc.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

freq, Amp_GR, Amp_modGR = np.loadtxt('data_amplitude_vs_freq.txt',delimiter=' ', unpack=True)


plt.figure()
plt.loglog(freq,Amp_GR,color='r',label='GR')
plt.loglog(freq,Amp_modGR,color='k',label='mod GR')
plt.xlabel('$f$ (Hz)',fontsize=14)
plt.ylabel('|h(f)|',fontsize=14)
plt.ylim(1e-25,(1e-22)/2)
plt.xlim(20,1000)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('Amplitude_birefringence_log_log_comparison.png')
plt.close()

