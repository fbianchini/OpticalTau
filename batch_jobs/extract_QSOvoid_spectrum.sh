#!/bin/bash

#SBATCH -p cloud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem 20000
#SBATCH --time=01:00:00
##SBATCH --mail-user=srinivasan.raghunathan@unimelb.edu.au
#SBATCH --mail-type=ALL

##module load Python/2.7.9-GCC-4.9.2-bare
##module load xcb-proto/1.11-intel-2016.u3-Python-2.7.9
module use /home/sri/modulefiles/
module load anaconda

cd ..
python extract_QSOpatch_spectrum.py 300 35 10

##for mass in $(seq 1.25 0.25 5.0)
##do
##        cmd="python s1_get_cov_matrices.py 25000 $mass 0.1 10 20160920/W_HU_NO_Wiever_filter/pol params.txt"
##        eval "$cmd" 
##done
#echo "Start - `date`"
##python s1_get_cov_matrices.py noofsims massval dxval snxval covfolder_val paramsfile_val
##echo "End - `date`"
