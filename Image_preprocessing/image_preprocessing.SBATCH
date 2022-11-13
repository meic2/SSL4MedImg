#!/bin/bash
#
#SBATCH --job-name=image_preprocessing
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --mem=32GB
#SBATCH --mail-type=END
#SBATCH --mail-user= ## Please add your email
#SBATCH --output=image_preprocessing.out

module purge
## Please change python file name in line 18

singularity exec \
	    --overlay /scratch/NetID/my_env/overlay-10GB-400K.ext3:ro \
	    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python filename.py"