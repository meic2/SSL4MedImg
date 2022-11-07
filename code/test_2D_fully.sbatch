#!/bin/bash
#
#SBATCH --job-name=99ptest1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=1:00:00
#SBATCH --mem=64GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mc8895@nyu.edu
#SBATCH --output=dermofit_ssl_99ptest1.out

module purge

singularity exec --nv\
	    --overlay /scratch/mc8895/overlay-25GB-500K-ssl4mis.ext3:ro \
	    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "source /ext3/env.sh; conda activate ssl; python test_2D_fully.py \
                    --root_path ../../dataset/Dermato_interpolated \
                    --exp Dermatomyositis/Cross_Teaching_CNN_Transformer_interpolated \
                    --saved_model_path /scratch/mc8895/SSL4SL/model/Dermatomyositis/Cross_Teaching_CNN_Transformer_interpolated_99p/unet/unet_best_model1.pth \
                    --labeled_num 99p \
                    --one_or_two 1 \
                    --data_class 3 \
                    --patch_size [480,480]"
