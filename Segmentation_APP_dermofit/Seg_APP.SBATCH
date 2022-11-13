#!/bin/bash
#
#SBATCH --job-name=Seg_APP_seed73_resnet18_unet_woAE
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=02:00:00
#SBATCH --mem=64GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user= ## Please add your email
#SBATCH --output=./output_file/Seg_APP_seed73_resnet18_unet_woAE.out

module purge

singularity exec --nv\
	    --overlay /scratch/netID/my_env/overlay-10GB-400K.ext3:ro \
	    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "source /ext3/env.sh; python Seg_APP.py --seed 73 --en resnet18 --unet"