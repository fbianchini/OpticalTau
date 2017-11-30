import numpy as np, os, time, pickle, gzip, sys

import os, sys, numpy as np

start, stop = int(sys.argv[1]), int(sys.argv[2])
wnoise      = int(sys.argv[3])

delta = 4
# start, stop = 0, 250
cutarr = np.arange(start,stop, delta)


extra = 'dl_sims_5deg_MASTER_dl20'
for c in cutarr:

	template = open('template.sh','r')
	op_file = '%s_%d_%d.sh' %(extra,c,c+delta)
	opfile = open(op_file,'w')
	for lines in template:
		opfile.writelines('%s\n' %lines.strip())

	cmdline = 'python extract_patches_spectra_sim.py %d %d %d' %(c, c+delta, wnoise)
	opfile.writelines('%s\n' %cmdline)

	opfile.close()
	template.close()#;quit()
	cmd = 'cd ..; sbatch batch_jobs/%s' %(op_file)#;quit()
	os.system(cmd)
	print cmd#;quit()


