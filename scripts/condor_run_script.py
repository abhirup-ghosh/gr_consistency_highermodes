import os
import numpy as np

M_list=[40.00,60.00,80.00,100.00,120.00,140.00,160.00,180.00,200.00]
#q_list=[0.11,0.12,0.14,0.17,0.20,0.25,0.33,0.50,0.67,1.00]
#q_map={0.11:9,0.12:8,0.14:7,0.17:6,0.20:5,0.25:4,0.33:3,0.50:2,0.67:1.5,1.00:1}
iota_list=[0.52,0.79,1.05,1.40,1.57]
iota_map={0.52:30,0.79:45,1.05:60,1.40:80,1.57:90}


for M in M_list:
  for iota in iota_list:
	data_dir = '/home/abhirup/Documents/Work/gr_consistency_highermodes/injections/9_param_runs_3det'
	out_dir = '/home/abhirup/Documents/Work/gr_consistency_highermodes/runs/9_param_runs_3det_complex_ampcorr/Mc_q_c1_diffM/M_%.2f_q_0.11_iota_%d'%(M, iota_map[iota])
	os.system('mkdir -p %s'%out_dir)

	sh_script = open(out_dir + "/condor_tgr.sh", "w")
	sub_script = open(out_dir + "/condor_tgr.sub", "w")
	

	sh_script.write("#!/bin/bash\n")
	sh_script.write("source ~/.bashrc\n")
	sh_script.write("source /home/abhirup/src/pycbc/bin/activate\n")
	sh_script.write("python /home/abhirup/Documents/Work/gr_consistency_highermodes/scripts/emcee_inference_tgr_complex_amplitude_correction_c1_tousif.py -H %s/GR_M_%.2f_q_0.11_iota_%.2f_flow_20Hz_H1.dat -L %s/GR_M_%.2f_q_0.11_iota_%.2f_flow_20Hz_L1.dat -V %s/GR_M_%.2f_q_0.11_iota_%.2f_flow_20Hz_V1.dat -i %s/GR_M_%.2f_q_0.11_iota_%.2f_flow_20Hz_ampcorr_initial.dat -o %s\n"%(data_dir, M, iota, data_dir, M, iota, data_dir, M, iota, data_dir, M, iota, out_dir))
	sh_script.close()

	sub_script.write("universe   = vanilla\n")
	sub_script.write("executable = %s/condor_tgr.sh\n"%out_dir)
	sub_script.write("output     = %s/condor_tgr.out\n"%out_dir)
	sub_script.write("error      = %s/condor_tgr.err\n"%out_dir)
	sub_script.write("log        = %s/condor_tgr.log\n"%out_dir)
	sub_script.write("request_cpus = 30\n")
	sub_script.write("queue\n")
	sub_script.close()

	os.system("chmod +x %s/condor_tgr.sh"%out_dir)
