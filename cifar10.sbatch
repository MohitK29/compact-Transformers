#!/bin/bash
# 
#SBATCH --nodes=1                       
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16                
#SBATCH --time=25:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=results
#SBATCH --output=cifar10.out
#SBATCH --gres=gpu:1


module purge
module load anaconda3/2020.07
eval "$(conda shell.bash hook)"
conda activate deep
python cvtTrain.py -c configs/datasets/cifar10.yml --model cct_7_3x1_32
