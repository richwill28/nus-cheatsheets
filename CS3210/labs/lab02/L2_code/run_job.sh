#!/bin/bash
#	CS3210 SBATCH Runner v1.0
#
#	You will primarily be changing the settings in GENERAL SETTINGS and SLURM SETTINGS.
#
#	USAGE: 		./run_job.sh <size> <threads>
#			Change the settings below this comment section (GENERAL and SLURM settings).

#	DESCRIPTION: 	Runs programs with sbatch while addressing a few usability issues.
#				- Lets you work in any directory you want, no need to work in the home directory / copy files to homedir
#				- Automatically copies specified code folders to NFS using rsync
#				- Changes to home before running to avoid Slurm crashing
#				- Executes the job: the job copies the code folder from NFS to a local working dir without using sbcast
#					- sbcast only works with single files, so we use srun
#				- The job then copies the output files (one each for slurm and the actual executable) back to NFS
#				- The runner then creates symlinks to the NFS logfiles in the user's submit directory
#				- It does some more utility things like handle some errors and print squeue/sacct output

# Exit when any command fails: remove this if necessary
set -e

##########################
# 1. GENERAL SETTINGS 	 #
##########################

# A meaningful name for this job
export SBATCH_JOB_NAME="mm1_job"

# You can change this to your username, but not necessary.
export USERNAME=$USER

# Change this to the directory with everything your job needs, will be copied to entirely to NFS!
export CODE_DIR=code/

# Change this to point at the main executable or shell script you want to run
# This is relative to CODE_DIR (the executable must be within CODE_DIR, or be accessible universally like `hostname`)
export EXECUTABLE=mm1.out
# Change this to change the arguments passed to the executable, comment out this line if there are no args
export EXECUTABLE_ARGS="${1} ${2}"

# Destination directory in NFS that your code directory is copied to, not necessary to change
export NFS_DIR=/nfs/home/$USERNAME/$SBATCH_JOB_NAME/

# Change this to your job file, we have provided one example.
# This job file must be inside the CODE_DIR!
export SBATCH_FILE="job.sh"

##########################
# 2. SLURM SETTINGS 	 #
##########################

## Even though it seems like these environment variables are not used anywhere,
## Slurm is actually using them automatically when we call sbatch.

# Uncomment this line to run your job on a specific partition
export SBATCH_PARTITION="i7-7700"

# Uncomment this line to run your job on a specific node (takes priority over partition)
# export SBATCH_WHICHNODE="soctf-pdc-006"

# How many nodes to run this job on
export SBATCH_NODES=1

# How many tasks the slurm controller should allocate resources for (leave at 1 for now)
export SBATCH_TASKS=1

# Memory required for each node
export SBATCH_MEM_PER_NODE="1G"

# Job time limit (00:10:00 --> 10 minutes)
export SBATCH_TIME_LIMIT="00:10:00"

# Job output and error names - these are relative to the local home directory
# Do NOT change this unless you change the copy step of the job script and the symlink step of this script
export SBATCH_OUTPUT="$SBATCH_JOB_NAME-%j.slurmlog"
export SBATCH_ERROR="$SBATCH_JOB_NAME-%j.slurmlog"

##################
# 3. EXECUTE 	 #
##################

# Copy all required code to the NFS directory (this overwrites the existing files if they are there)
[[ -e $CODE_DIR ]] || {
	echo "!!! Runner: CODE_DIR $CODE_DIR does not exist, quitting..."
	exit 1
}
echo -e "\n>>> Runner: Changing directory to $CODE_DIR\n"
INITIAL_DIR=$PWD
cd $CODE_DIR
echo -e "\n>>> Runner: Synchronizing files between local directory (./) and remote directory ($NFS_DIR)\n"
mkdir -p $NFS_DIR/
rsync -av --progress . "$NFS_DIR/"

# Change to local home dir to execute batch script
cd /home/$USERNAME/

# Prepare to execute the job file
SBATCH_FILE_FULL="$NFS_DIR/$SBATCH_FILE"
[[ -f $SBATCH_FILE_FULL ]] || {
	echo "!!! Runner: sbatch file $SBATCH_FILE_FULL does not exist, quitting..."
	exit 1
}
echo -e "\n>>> Runner: Executing $SBATCH_FILE_FULL with Slurm\n"

# Execute the sbatch command with all arguments
if [[ ! -z ${SBATCH_WHICHNODE} ]]; then
	set -x
	jobid=$(sbatch \
		--nodes=$SBATCH_NODES \
		--ntasks=$SBATCH_TASKS \
		--mem=$SBATCH_MEM_PER_NODE \
		--time=$SBATCH_TIME_LIMIT \
		--output=$SBATCH_OUTPUT \
		--error=$SBATCH_ERROR \
		-w $SBATCH_WHICHNODE \
		--parsable \
		$SBATCH_FILE_FULL)
	set +x
else
	set -x
	jobid=$(sbatch \
		--nodes=$SBATCH_NODES \
		--ntasks=$SBATCH_TASKS \
		--mem=$SBATCH_MEM_PER_NODE \
		--time=$SBATCH_TIME_LIMIT \
		--output=$SBATCH_OUTPUT \
		--error=$SBATCH_ERROR \
		--parsable \
		$SBATCH_FILE_FULL)
	set +x
fi
echo -e "\n>>> Runner: Submitted slurm job with ID $jobid"

##########################
# 4. CLEANUP AND UTIL 	 #
##########################

# Change to job submission directory
cd $INITIAL_DIR >/dev/null

# Create symlinks in current directory to the nfs logfiles
echo -e "\n>>> Runner: creating symlinks to NFS working directory and job logs"
ln -nsf $NFS_DIR ./nfs_dir
ln -nsf $NFS_DIR/$SBATCH_JOB_NAME-$jobid.slurmlog ./latest_slurm_"${1}"_"${2}"_log.slurmlog
ln -nsf $NFS_DIR/$SBATCH_JOB_NAME-$jobid.out ./latest_program_"${1}"_"${2}"_log.out

echo -e "\n>>> Runner: printing queue and account status"

# Show queue just to confirm job status
echo -e "\n>>> Runner: Your job's current status in the Slurm queue:\n"
squeue -u $USERNAME

# Sleep to let the job hit the accounting system
echo -e "\nRun the command:\n\tsacct -j $jobid --format=JobID,Start,End,Elapsed,NCPUS,NodeList,NTasks,State\nto see job status"

# Finish
echo -e "\n>>> Runner:\n\t- Finished submitting job.\n\t- Please wait for job to end (check squeue or use sacct command above).\n\t- Then, symlinks (shortcuts) to your:\n\t\t- program output (latest_program_log.out)\n\t\t- slurm log (latest_slurm_log.slurmlog), and\n\t\t- NFS folder with all your logs (nfs_dir)\n\t\twill be active in this folder."
