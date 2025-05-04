#!/bin/bash
#SBATCH -J train_eval_job
#SBATCH -p all
#SBATCH --qos=comp579-1gpu-12h
#SBATCH -N 1
#SBATCH -c 16                  # Max allowed by QOS (not node limit)
#SBATCH --mem=120G             # Conservative estimate (node has 500GB)
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu-grad-02
#SBATCH -t 12:00:00
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err

# Load CUDA and environment
export PATH=$HOME/cuda-10.1/bin:$PATH
export LD_LIBRARY_PATH=$HOME/cuda-10.1/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=$HOME/cuda-10.1
source ~/.bashrc

python "og_train_eval_stage_2.py" 