#!/bin/bash
#
#SBATCH --job-name=request_gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=05:00:00
#SBATCH --mem=96GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lc4866@nyu.edu
#SBATCH --output=request_gpu.out

sleep 8h



