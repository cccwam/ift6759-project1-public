#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10000M
#SBATCH --reservation=IFT6759_2020-01-10

# Summary:
#   A template for launching a batch job to execute code from
#   this repository on a k80 gpu node.
# Setup and example usage:
#   1. Copy this file to your helios' home folder ~/
#   2. Perform necessary modifications to the SBATCH parameters
#      at the top of this script.
#   3. Configure LOCAL_GIT_REPO_FOLDER and COMMAND below.
#   4. Execution: sbatch ~/sbatch_template.sh

LOCAL_GIT_REPO_FOLDER="~/ift6759-project1"
COMMAND="python train.py"

shopt -s extglob # enables cp to copy with exclusions !(...)
cp !(venv) -a $LOCAL_GIT_REPO_FOLDER/. $SLURM_TMPDIR/
shopt -u extglob

module load python/3.7
virtualenv --no-download $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate
pip install --no-index --upgrade pip
pip install -r requirements.txt

eval "$COMMAND"
