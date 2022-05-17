#!/bin/bash
# 
#SBATCH --nodes=1                       
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16                
#SBATCH --time=40:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=results
#SBATCH --output=cvtgan.out
#SBATCH --gres=gpu:1


module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate deep
CUDA_VISIBLE_DEVICES=0 \
python cvtGanTrain.py \
  -cfg=cvtgan_cifar10.yaml \
  -dataset=cifar10 \
  -batch_size=32 \