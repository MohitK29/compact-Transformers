#!/bin/bash
# 
#SBATCH --nodes=1                       
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16                
#SBATCH --time=20:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=results
#SBATCH --output=exp5.out
#SBATCH --gres=gpu:1


module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate deep
python train.py -c configs/datasets/imagenet.yml --model cct_14_7x2_224