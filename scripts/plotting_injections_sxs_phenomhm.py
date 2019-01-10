import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

postlocdir_root = '../runs/SXS_four_modes_20181223_cosiota_tgr'

iota_list = [0.00, 0.79, 1.05, 1.57]
psi_list = [0.00, -1.57]

postloc_list = ['/BBH_M_70.00_iota_1.57_psi_0.00_t0_0_phenomhm/BBH_M_70.00_iota_1.57_psi_0.00_t0_0_phenomhm_data.dat', '/BBH_M_70.00_iota_1.57_psi_0.00_t0_0_sxs/BBH_M_70.00_iota_1.57_psi_0.00_t0_0_sxs_data.dat', '/NSBH_M_70.00_iota_1.57_psi_0.00_t0_0_sxs/NSBH_M_70.00_iota_1.57_psi_0.00_t0_0_sxs_data.dat']
label_list = ['phenomhm', 'bbh-sxs', 'nsbh-sxs']
color_list = ['r', 'g', 'k']

for iota in iota_list:
  for psi in psi_list:

    plt.figure(figsize=(10,10))
    
    postloc_list = ['/BBH_M_70.00_iota_%.2f_psi_%.2f_t0_0_phenomhm/BBH_M_70.00_iota_%.2f_psi_%.2f_t0_0_phenomhm_data.dat'%(iota, psi, iota, psi), '/BBH_M_70.00_iota_%.2f_psi_%.2f_t0_0_sxs/BBH_M_70.00_iota_%.2f_psi_%.2f_t0_0_sxs_data.dat'%(iota, psi, iota, psi), '/NSBH_M_70.00_iota_%.2f_psi_%.2f_t0_0_sxs/NSBH_M_70.00_iota_%.2f_psi_%.2f_t0_0_sxs_data.dat'%(iota, psi, iota, psi)]

    for (idx, postloc) in enumerate(postloc_list):

	f, data_r, data_i, psd = np.genfromtxt(postlocdir_root + postloc,unpack=True)

	data = data_r + 1j*data_i

	plt.subplot(221)
	plt.loglog(f, abs(data), ls='dashed', color=color_list[idx])
	plt.xlim(20, 999)
	plt.xlabel('frequency  (Hz)')
	plt.ylabel('amplitude')
	plt.subplot(222)
        plt.loglog(f, np.unwrap(np.angle(data)), ls='dashed', label=label_list[idx], color=color_list[idx])
	plt.legend(loc='best')
	plt.ylabel('phase')
	plt.xlim(20, 999)
	plt.subplot(223)
        plt.loglog(f, np.real(data), ls='dashed', color=color_list[idx])
	plt.ylabel('Re(data)')
	plt.xlim(20, 100)
	plt.subplot(224)
        plt.loglog(f, np.imag(data), ls='dashed', color=color_list[idx])
	plt.ylabel('Im(data)')
	plt.xlim(20, 100)
	plt.hold(True)

    plt.savefig(postlocdir_root + '/data_comparison_iota_%.2f_psi_%.2f.png'%(iota, psi))
    plt.close()
