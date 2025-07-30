#!/bin/bash
#SBATCH --job-name=spectrum
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem=250G
#SBATCH --chdir=/gscratch/mccoy/mbrosen/dms/dmsklone
#SBATCH --partition=cpu-g2
#SBATCH --account=chem
#SBATCH --ntasks-per-node=1

# Set up email notifications
#SBATCH --mail-type=END,FAIL  # Notify when the job ends or fails
#SBATCH --mail-user=mbrosen@uw.edu # Your email address
# load Gaussian environment
source ~/.bashrc
conda activate mbrosen

# Set the PYTHONPATH to the current directory to ensure 'Code' is found
export PYTHONPATH=$PYTHONPATH:/gscratch/mccoy/mbrosen/dms/dmsklone

# debugging information
echo "**** Job Debugging Information ****"
echo "This job will run on $SLURM_JOB_NODELIST"
echo ""
echo "ENVIRONMENT VARIABLES"
set
echo "**********************************************"

# run the Python script
python ./J40max.py
