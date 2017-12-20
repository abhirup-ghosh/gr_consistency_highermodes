import matplotlib
#matplotlib.use("Pdf")
import numpy as np
import matplotlib.pyplot as plt
import bayesian as ba
import imrtestgr as tgr

data=np.genfromtxt('data_mod_gr.txt',delimiter=' ')

delta_mc_2=data[:,0] # delta mc in 2 independent variable case
delta_q_2=data[:,1]  # delta q  in 2 independent variable case
delta_mc_1=data[:,2] # delta mc in 1 independent variable case
delta_q_1=data[:,3]  # delta q  in 1 independent variable case

#######
#1d-histograms of delta mc ,delta q - 1 independent variable case 
#######
N_bins = 35
dmc_min = min(delta_mc_1)
dmc_max = max(delta_mc_1)
dq_min = min(delta_q_1)
dq_max = max(delta_q_1)
dmc_bins = np.linspace(dmc_min, dmc_max, N_bins)
dq_bins = np.linspace(dq_min, dq_max, N_bins)
dmc_hist_1,dmc_edges_1 = np.histogram(delta_mc_1,N_bins,normed=True)
dq_hist_1,dq_edges_1 = np.histogram(delta_q_1,bins=dq_bins,normed=True)


#######
#1d-histograms of delta mc ,delta q - 2 independent variable case 
#######

dmc_min = min(delta_mc_2)
dmc_max = max(delta_mc_2)
dq_min = min(delta_q_2)
dq_max = max(delta_q_2)
dmc_bins = np.linspace(dmc_min, dmc_max, N_bins)
dq_bins = np.linspace(dq_min, dq_max, N_bins)
dmc_hist_2,dmc_edges_2 = np.histogram(delta_mc_2,N_bins,normed=True)
dq_hist_2,dq_edges_2 = np.histogram(delta_q_2,bins=dq_bins,normed=True)


#######
#2d-histograms of delta mc ,delta q - 2 independent variable case 
#######

N_bins=50
dmc_min = min(delta_mc_2)
dmc_max = max(delta_mc_2)
dq_min = min(delta_q_2)
dq_max = max(delta_q_2)
dmc_bins_2 = np.linspace(dmc_min, dmc_max, N_bins)
dq_bins_2 = np.linspace(dq_min, dq_max, N_bins)

P_dmc_dq_2, dmc_bins_2, dq_bins_2 = np.histogram2d(delta_mc_2, delta_q_2, bins=(dmc_bins_2, dq_bins_2), normed=True)
P_dmc_dq_2 = P_dmc_dq_2.T
s1 = ba.nsigma_value(P_dmc_dq_2, 0.68) # 1 sigma level of 2d histogram of delta mc ,delta q - 2 independent variable case
s2 = ba.nsigma_value(P_dmc_dq_2, 0.95) # 2 sigma level of 2d histogram of delta mc ,delta q - 2 independent variable case


plt.figure()
plt.subplot2grid((3,3), (0,0), colspan=2)
plt.plot(dmc_edges_1[:-1],dmc_hist_1,'b', lw=1)
plt.plot(dmc_edges_2[:-1],dmc_hist_2,'k', lw=1)
plt.axvline(0.,color='k', lw=0.6,linestyle='-.')
plt.ylabel('$P(\Delta M_c)$',fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)


plt.subplot2grid((3,3), (1,0), colspan=2, rowspan=2)
plt.pcolormesh(dmc_bins_2[:-1],dq_bins_2[:-1],P_dmc_dq_2, cmap='YlOrBr')
plt.contour(dmc_bins_2[:-1],dq_bins_2[:-1],tgr.gf(P_dmc_dq_2), levels=(s2,s1), linewidths=(1,1.5))
plt.plot(0, 0, 'w+', ms=12, mew=2) # for mod gr make the marker k+ , for gr make the marker w+
plt.xlabel('$\Delta M_c (M_\odot)$',fontsize=14)
plt.ylabel('$\Delta q$',fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)


plt.subplot2grid((3,3), (1,2), rowspan=2)
plt.plot(dq_hist_2, dq_edges_2[:-1],'k', lw=1)
plt.plot(dq_hist_1, dq_edges_1[:-1],'b', lw=1)
plt.axhline(0.,color='k', lw=0.6,linestyle='-.')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('$P(\Delta q)$',fontsize=14)

plt.tight_layout()
plt.savefig('triangle_plot_gr.png')
plt.close()

