#!/bin/bash
#PBS -N cs.n1.25.t2.data
#PBS -o csout.n1.25.t2.log
#PBS -e cserr.n1.25.t2.log
#PBS -l select=1:ncpus=16:mem=32gb
#PBS -l walltime=12:30:00
#PBS -q large
echo "Starting at "`date`
cd $PBS_O_WORKDIR
cd ..
export PATH=$PATH:/home/apps/GnuParallel/bin
cd $PBS_WORKDIR
parallel --jobs 16 < $PBS_O_WORKDIR/ipjobs_cs_n1-25.t02.sh
echo "Finished at "`date`