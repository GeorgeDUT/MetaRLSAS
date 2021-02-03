#!/bin/bash
#SBATCH -o job-meta-rl.%j.%N.out
#SBATCH --partition=C032M0128G
#SBATCH --qos=high
#SBATCH -A hpc0006178148
#SBATCH -J micra-new
#SBATCH --get-user-env
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --mail-type=end
#SBATCH --time=120:00:00

module load anaconda/3-4.4.0.1
conda info -e
source activate rl-basic
python ~/MetaRLSAS/test-my-plus.py --config ~/MetaRLSAS/mdp-deterministic/config.json --policy ~/MetaRLSAS/mdp-deterministic/policy.th --output ~/MetaRLSAS/mdp-deterministic/results.npz --meta-batch-size 10 --num-batches 5 --num-workers 8
