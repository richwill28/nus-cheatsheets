#!/bin/bash
#SBATCH --job-name=example
#SBATCH --nodes=1
#SBATCH --ntasks=1

echo "Running job $SLURM_JOB_NAME on $(hostname) at $(date)!"

# Sanity checks
LOCAL_DIR="/home/$USERNAME/$SLURM_JOB_NAME/"
[[ ! -z "$NFS_DIR" || ! -z "$EXECUTABLE" ]] || {
	echo "This script can only be run with our driver as it sets env vars, we are missing NFS_DIR=$NFS_DIR EXECUTABLE=$EXECUTABLE"
	exit 1
}

# Broadcast all our files to local directories. sbcast can only do this for single files, so we use srun.
echo "Synchronizing data from $NFS_DIR/ to $LOCAL_DIR"
srun mkdir -p $LOCAL_DIR/
srun rsync -av --progress --delete "$NFS_DIR" "$LOCAL_DIR/"

##################################################################
# Change the code under these if branches	 		 #
# if you want to run anything other than `perf stat $EXECUTABLE` #
##################################################################

# Check if the user meant to run an executable inside the local dir or a global executable
# Then, run actually run it.
PROGRAM_OUTFILE="${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
if [[ -f "$LOCAL_DIR/$EXECUTABLE" ]]; then
	echo "Running executable from local folder: $LOCAL_DIR/$EXECUTABLE $EXECUTABLE_ARGS"
	srun perf stat -r 5 -ddd "$LOCAL_DIR/$EXECUTABLE" $EXECUTABLE_ARGS >$PROGRAM_OUTFILE 2>&1
	# srun perf stat -r 5 -e task-clock,context-switches,cpu-migrations,page-faults,cycles,instructions,branches,branch-misses,fp_arith_inst_retired.scalar_single "$LOCAL_DIR/$EXECUTABLE" $EXECUTABLE_ARGS >$PROGRAM_OUTFILE 2>&1
else
	echo "Running executable from anywhere in PATH: $EXECUTABLE $EXECUTABLE_ARGS"
	srun perf stat -r 5 -ddd $EXECUTABLE $EXECUTABLE_ARGS >$PROGRAM_OUTFILE 2>&1
	# srun perf stat -r 5 -e task-clock,context-switches,cpu-migrations,page-faults,cycles,instructions,branches,branch-misses,fp_arith_inst_retired.scalar_single $EXECUTABLE $EXECUTABLE_ARGS >$PROGRAM_OUTFILE 2>&1
fi

# This is useful to know when the job ends
echo "Job $SLURM_JOB_NAME ended on $(hostname) at $(date)"

##################################
# Update the code below		 #
# if you have any more logfiles  #
##################################

# Copy our logfiles from node to shared drive
echo "Copying logfile from ${SLURM_JOB_NAME}-${SLURM_JOB_ID}.slurmlog to $NFS_DIR/"
cp -r "${SLURM_JOB_NAME}-${SLURM_JOB_ID}.slurmlog" "$NFS_DIR/"
echo "Copying program output from ${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out to $NFS_DIR/"
cp -r $PROGRAM_OUTFILE "$NFS_DIR/"
