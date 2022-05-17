#!/bin/bash
# 
#SBATCH --nodes=1                       
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16                
#SBATCH --time=40:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=results
#SBATCH --output=vitgan.out
#SBATCH --gres=gpu:1


module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate deep
python cvtGenerate.py \
  -cfg=cifar10_cifar10.yaml \
  -num_out_images=16 \
  -out_folder=cvtGan1 \
  -pretrained=/path/to/pretrained/model/cvtgan_cifar10 