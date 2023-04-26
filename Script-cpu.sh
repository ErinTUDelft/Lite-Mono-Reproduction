#!/bin/bash

#SBATCH --job-name="Lite-Mono-CPU"
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=8GB
#SBATCH --account=education-ae-msc-ae
module load 2022r2
module load openmpi python/3.8.12 py-pip
module unload py-torch
srun pip3 list 
srun python3 train.py --no_cuda --data_path "/scratch/elucassen/kitti_data" --model_name Erin_train_CPU_1 --batch_size 6 >Erin.cpu.$$.log 2>Erin.cpu.$$.err
