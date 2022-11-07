#!/bin/bash
#
#SBATCH --job-name=TilingOnly_50p
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=6:00:00
#SBATCH --mem=64GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lc4866@nyu.edu
#SBATCH --output=CT_Between_CNN_Transformer_TilingOnly_label7.out

module purge

singularity exec --nv\
     --overlay /scratch/lc4866/overlay-example/overlay-50G-10M.ext3:ro \
     /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
     /bin/bash -c "source /ext3/env.sh; conda activate ssl; bash train_Dermatomyositis.sh"