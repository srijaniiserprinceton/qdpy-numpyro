#!/bin/bash
#SBATCH --job-name=rot-qdpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --mem=150G
#SBATCH --time=130:00:00          # total run time limit (HH:MM:SS)

# commands commented out
# SBATCH --cpus-per-task=40       # cpu-cores per task (>1 if multi-threaded tasks)
# SBATCH --mem-per-cpu=18G        # memory per cpu-core (4G is default)
# SBATCH --mail-type=begin        # send email when job begins
# SBATCH --mail-type=end          # send email when job ends
# SBATCH --mail-user=sbdas@princeton.edu

module purge
module load anaconda3
conda activate jax-gpu

n0=0
lmin=195
lmax=245
maxiter=2500
/home/sk0551/.conda/envs/jax-gpu/bin/python qdpy_jax/pyrotry.py --n0 ${n0} --lmin ${lmin} --lmax ${lmax} --maxiter ${maxiter} > jobscripts/slurm.out 2> jobscripts/slurm.err
