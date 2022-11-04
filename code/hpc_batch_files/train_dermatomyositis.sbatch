#!/bin/bash
#
#SBATCH --job-name=dermatomyositis_train_ssl
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=6:00:00
#SBATCH --mem=64GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mc8895@nyu.edu
#SBATCH --output=dermatomyositis_train_ssl.out

module purge

singularity exec --nv\
	    --overlay /scratch/mc8895/overlay-25GB-500K-ssl4mis.ext3:ro \
	    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "source /ext3/env.sh; conda activate ssl; python train_cross_teaching_between_cnn_transformer_2D.py --root_path=Dermatomyositis --exp=Dermatomyositis/Cross_Teaching_Between_CNN_Transformer"