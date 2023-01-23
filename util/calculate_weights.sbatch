#!/bin/bash
#
#SBATCH --job-name=cw
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --time=00:10:00
#SBATCH --mem=32GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user= ## Please add your email
#SBATCH --output=calculate_weight.out

singularity exec\
	    --overlay /scratch/NetID/my_env/overlay-10GB-400K.ext3:ro \
	    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python calculate_weights.py"