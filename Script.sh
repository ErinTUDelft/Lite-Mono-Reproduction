#!/bin/bash

#SBATCH --job-name="Lite-Mono"
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=1GB


module load 2022r2

module load openmpi
module load python
module load py-numpy/1.19.5
module load py-mpi4py

module load six
module load linear-warmup-cosine-annealing-warm-restarts-weight-decay 
module load py-torch
module load py-torchvision
module load py-tensorboardX
module load timm


cd /scratch/elucassen/Lite-Mono-Reproduction 
git pull 
srun python train.py --data_path "/scratch/elucassen/kitti_data" --model_name Erin_train_Saturday_1 --batch_size 16
