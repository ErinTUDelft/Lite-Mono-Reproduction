#!/bin/bash

#SBATCH --job-name="Lite-Mono"
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=1GB
#SBATCH --account=education-ae-msc-ae


module load 2022r2

module load openmpi
module load python
module load py-numpy
module load py-mpi4py

module load py-six
module load py-linear-warmup-cosine-annealing-warm-restarts-weight-decay 
module load py-torch
module load py-torchvision
module load py-tensorboardX
module load py-timm
module load py-imageio
module load py-scikit-image
module load py-scipy
module load py-tifffile
module load py-PyWavelets
module load py-requests
module load py-PyYAML
module load py-setuptools
module load py-tqdm
module load py-typing_extensions
module load py-urllib3

module load py-certifi                                                
module load py-charset-normalizer                                   
module load py-contourpy                                              
module load py-cycler                                                 
module load py-filelock                                               
module load py-fonttools                                              
module load py-huggingface-hub                                        
module load py-idna                                                  

module load py-importlib-resources                                      
module load py-install                                                 
module load py-kiwisolver                                            
module load py-lazy-loader                                            
module load py-matplotlib                                              
module load py-networkx                                               

module load py-packaging                                               
module load py-Pillow                                                  
module load py-pyparsing                                               
module load py-python-dateutil                                          


srun python train.py --data_path "/scratch/elucassen/kitti_data" --model_name Erin_train_Saturday_1 --batch_size 16 >Erin.$$.log 2>Erin.$$.err
