import numpy as np, os, time, pickle, gzip, sys

import os, sys, numpy as np

delta = 4
start, stop = 0, 250#1.5e7,2.1e7#e0, 2e6#7e6, 1e7#, 2.1e7+delta
cutarr = np.arange(start,stop, delta)

#extra = 'Wscos'
#extra='smica'
extra = 'dl_5deg_MASTER_dl20'
for c in cutarr:

	template = open('template.sh','r')
	op_file = '%s_%d_%d.sh' %(extra,c,c+delta)
	opfile = open(op_file,'w')
	for lines in template:
		opfile.writelines('%s\n' %lines.strip())

	cmdline = 'python extract_patches_spectra.py %d %d' %(c, c+delta)
	opfile.writelines('%s\n' %cmdline)

	opfile.close()
	template.close()#;quit()
	cmd = 'cd ..; sbatch batch_jobs/%s' %(op_file)#;quit()
	os.system(cmd)
	print cmd#;quit()


