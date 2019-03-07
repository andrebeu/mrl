#!/bin/bash

printf "\n\n -- cluster_loop --\n"

stamp=$(date +"%m%d%H%M%S")
wd_dir="/tigress/abeukers/wd/cswNets"

declare -a stsize_arr=(25 50 75)
declare -a lr_arr=(1 10 100 1000 5 5000)
declare -a gamma_arr=(5 10 20 30 70 80 90 95)

## now loop through the above array
for i in {1..5}; do 
	for stsize in "${stsize_arr[@]}"; do 
		for lr in "${lr_arr[@]}"; do 
			for gamma in "${gamma_arr[@]}"; do 
				sbatch ${wd_dir}/gpu_jobsub.cmd "${stsize}" "adam" "${lr}" "${gamma}" 
				sbatch ${wd_dir}/gpu_jobsub.cmd "${stsize}" "rms" "${lr}" "${gamma}" 
			done
		done
	done
done
