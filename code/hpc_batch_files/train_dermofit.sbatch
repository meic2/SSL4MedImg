#!/bin/bash
#
#SBATCH --job-name=dermo35_train_ssl
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=6:00:00
#SBATCH --mem=64GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mc8895@nyu.edu
#SBATCH --output=dermofit_ssl_35.out

module purge

singularity exec --nv\
	    --overlay /scratch/mc8895/overlay-25GB-500K-ssl4mis.ext3:ro \
	    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "source /ext3/env.sh; conda activate ssl; python train_cross_teaching_between_cnn_transformer_2D.py \
            --labeled_num=35 \
            --root_path=Dermofit  \
            --exp=Dermofit/Cross_Teaching_CNN_Transformer"