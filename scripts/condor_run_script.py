import os

pol_list=[-3.14,-1.57,0.00]
iota_list=[0.00,0.79,1.57]
cbc_list = ['NSBH', 'BBH']

for cbc in cbc_list:
  for iota in iota_list:
    for pol in pol_list:

	file_tag = '%s_iota_%.3f_pol_%.3f_Mtot_70_flow15'%(cbc, iota, pol)

	sh_script = open("/home/abhirup/Documents/Work/gr_consistency_highermodes/condor_scripts/%s.sh"%file_tag, "w")
	sub_script = open("/home/abhirup/Documents/Work/gr_consistency_highermodes/condor_scripts/%s.sub"%file_tag, "w")
	

	sh_script.write("#!/bin/bash\n")
	sh_script.write("source ~/.bashrc\n")
	sh_script.write("source /home/abhirup/src/pycbc/bin/activate\n")
	sh_script.write("python /home/abhirup/Documents/Work/gr_consistency_highermodes/scripts/emcee_inference_tgr_abhi.py -d /home/abhirup/Documents/Work/gr_consistency_highermodes/injections/modGR_simulations/%s_data.dat -o /home/abhirup/Documents/Work/gr_consistency_highermodes/runs/modGR_simulations/%s -i /home/abhirup/Documents/Work/gr_consistency_highermodes/injections/modGR_simulations/%s_initial.dat --save-incremental-progress False\n"%(file_tag,file_tag,file_tag))
	sh_script.close()

	sub_script.write("universe   = vanilla\n")
	sub_script.write("executable = %s.sh\n"%file_tag)
	sub_script.write("output     = %s.out\n"%file_tag)
	sub_script.write("error      = %s.err\n"%file_tag)
	sub_script.write("log        = %s.log\n"%file_tag)
	sub_script.write("request_cpus = 30\n")
	sub_script.write("queue\n")
	sub_script.close()

	os.system("chmod +x /home/abhirup/Documents/Work/gr_consistency_highermodes/condor_scripts/%s.sh"%file_tag)
