#!/bin/bash
#
#SBATCH --job-name=test
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:10:00
#SBATCH --mem=32GB
#SBATCH --mail-type=END
#SBATCH --mail-user= ## Please add your email
#SBATCH --output=test.out

module purge
## Please change python file name in line 18

singularity exec --nv \
	    --overlay /scratch/ssc10020/overlay-50G-10M.ext3:ro \
	    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "source /ext3/miniconda3/bin/activate; conda activate ssl; \
        python test_2D.py \
--exp 'Dermatomyositis_prototype/test' \
--one_or_two 2 \
--data_class 2 \
--labeled_num 30 \
--saved_model_path ../model/Dermatomyositis_prototype/test_30p/unet/unet_best_model2.pth \
--batch_size 16" 