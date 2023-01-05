#!/bin/bash
#SBATCH --job-name=cond
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1gb
#SBATCH --time=00:00:30
#SBATCH --output=cond_%j.log
#SBATCH --error=cond_%j.log

echo "Running cond job!"
echo "We are running on $(hostname)"

# Broadcast our executable to all allocated nodes
sbcast /nfs/home/$USER/cond /home/$USER/cond

# Actual job
/home/$USER/cond

# This is useful to know when the job ends
date

# Copy our logfile from node to shared drive
cp "cond_$SLURM_JOB_ID.log" "/nfs/home/$USER/"
