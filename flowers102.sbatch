#!/bin/bash
# 
#SBATCH --nodes=1                       
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16                
#SBATCH --time=5:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=results
#SBATCH --output=exp1.out
#SBATCH --gres=gpu:1


module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate deep
python train.py -c configs/datasets/flowers102.yml --model cct_7_7x2_224_sine 
