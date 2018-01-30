"""
This file is to plot 90% widths
"""

import matplotlib
import numpy as np
import matplotlib.pyplot as plt


q, i_30, i_45, i_60, i_80, i_90 = np.loadtxt('90_percent_width_dM_t.txt',delimiter=' ', unpack=True)# This file consists the data of 90% widths of (\Delta M /M) where M is total mass. This data is for inclination angles = 30,45,60,80,90 degrees having common q=2,3,4,5,6,7,8,9. The first column is all the q and i_angle is 90% widths for that angle and corresponding mass ratio  

plt.figure()
plt.semilogy(q,i_30,'r',label='$\iota=30^{\circ}$')
plt.semilogy(q,i_45,'g',label='$\iota=45^{\circ}$')
plt.semilogy(q,i_60,'b',label='$\iota=60^{\circ}$')
plt.semilogy(q,i_80,'brown',label='$\iota=80^{\circ}$')
plt.semilogy(q,i_90,'k',label='$\iota=90^{\circ}$')
plt.xlabel('Mass ratio(q)',fontsize=12)
plt.ylabel('90% width of $\Delta M/M$',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('dM_spread_vs_q.png')
plt.close()


