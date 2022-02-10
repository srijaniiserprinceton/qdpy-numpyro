#!/bin/bash
#SBATCH --job-name=dpt_newton
#SBATCH --nodes=1                                                                            
#SBATCH --ntasks-per-node=40                                                                
#SBATCH --mem=150G                                                                           
#SBATCH --time=02:00:00          # total run time limit (HH:MM:SS) 
echo "Starting at "`date`
module purge
module load anaconda3
conda activate jax-gpu2
parallel --jobs 20 < ipjobs_dpt_rls.sh
echo "Finished at "`date`
