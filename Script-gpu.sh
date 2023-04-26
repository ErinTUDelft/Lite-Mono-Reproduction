#!/bin/bash

#SBATCH --job-name="Lite-Mono-GPU"
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#S-COMMENT-BATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --account=education-ae-msc-ae
#SBATCH --mem-per-cpu=8GB
module load 2022r2
module load openmpi python/3.8.12 py-pip
module unload py-torch

srun pip3 list >Erin.GPU.piplist.$$.log

previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

srun python3 train.py --data_path "/scratch/elucassen/kitti_data" --model_name Erin_train_GPU_$$_1 --batch_size 6 >Erin.GPU.$$.log 2>Erin.GPU.$$.err

/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous" >Erin.GPU.nv.$$.log 2>Erin.GPU.nv.$$.err
