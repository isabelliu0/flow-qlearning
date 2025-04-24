#!/bin/bash
#SBATCH --job-name=fqltrain # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1             # total number of tasks across all nodes
#SBATCH --mem-per-cpu=40G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=vc9839@princeton.edu
#SBATCH --output=/scratch/gpfs/vc9839/flow-qlearning/log.out
#SBATCH --error=/scratch/gpfs/vc9839/flow-qlearning/err.out

cd /scratch/gpfs/vc9839/flow-qlearning

source env.sh

python3 main.py
