#!/bin/bash
#SBATCH --job-name=basic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=1gb
#SBATCH --time=00:00:30
#SBATCH --output=basic_%j.log
#SBATCH --error=basic_%j.log

echo "Running basic job!"
echo "We are running on $(hostname)"

# Actual "job"
sleep 5;

# This is useful to know when the job ends
date;

# Copy our logfile from node to shared drive
cp "basic_$SLURM_JOB_ID.log" "/nfs/home/$USER/"
