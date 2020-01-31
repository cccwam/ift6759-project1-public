#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10000M
#SBATCH --reservation=IFT6759_2020-01-10

# Summary:
#   A template for launching a batch job to execute code from
#   this repository on a k80 gpu node.
# Pre-requisites:
#   1. Clone this git repository to your helios home folder.
#   2. Copy this file to your helios' home folder ~/
#   3. Perform necessary modifications to the SBATCH parameters
#      at the top of this script.
#   4. Edit the following uppercase variables to suit your needs.
# Example usage:
#   sbatch ~/sbatch_template.sh

LOCAL_GIT_REPO_FOLDER=~/ift6759-project1
VENV_FOLDER=venv
COMMAND="python train.py"

cd $SLURM_TMPDIR/
cp -a $LOCAL_GIT_REPO_FOLDER/. $SLURM_TMPDIR/
module load python/3.7.4
module load hdf5-mpi/1.10.3
virtualenv --no-download $SLURM_TMPDIR/$VENV_FOLDER
source $SLURM_TMPDIR/$VENV_FOLDER/bin/activate
pip install --no-index --upgrade pip -r requirements.txt
$COMMAND
