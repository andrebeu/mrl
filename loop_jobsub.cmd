#!/bin/bash

printf "\n\n -- cluster_loop --\n"

#
wd_dir="/tigress/abeukers/wd/mrl" 
#

declare -a stsize_arr=(20 25 30 35 40)
# declare -a lr_arr=(1 5 10 50 100 500 1000 5000 10000 50000)
declare -a lr_arr=(100 500 1000)
declare -a gamma_arr=(50 60 70 80 90)

## now loop through the above array
for i in {1..5}; do 
	for stsize in "${stsize_arr[@]}"; do 
		for lr in "${lr_arr[@]}"; do 
			for gamma in "${gamma_arr[@]}"; do 
				sbatch ${wd_dir}/gpu_jobsub.cmd "${stsize}" "adam" "${lr}" "${gamma}" "${i}"
				sbatch ${wd_dir}/gpu_jobsub.cmd "${stsize}" "rms" "${lr}" "${gamma}" "${i}"
			done
		done
	done
done
