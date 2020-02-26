#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:k80:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10000M
# -SBATCH --reservation=IFT6759_2020-01-10

# Summary:
#   A template for launching a batch job to execute code from
#   this repository on a k80 gpu node.
# Pre-requisites:
#   1. Ensure you have internet access (be on a login node).
#   2. Clone this git repository to your helios home folder.
#   3. Create a virtual environment in your home folder:
#      ~/ift6759-project1/tools/create_venv.sh
#   4. Copy the sbatch_template.sh to your home folder:
#      cp ~/ift6759-project1/tools/sbatch_template.sh ~/
#   5. In ~/sbatch_template.sh, perform necessary modifications
#      to the SBATCH parameters at the top of this script and
#      the python command at the bottom.
#   6. Edit LOCAL_GIT_REPO_FOLDER and LOCAL_VENV_FOLDER as necessary.
# Example usage:
#   sbatch ~/sbatch_template.sh

LOCAL_GIT_REPO_FOLDER=~/ift6759-project1
LOCAL_VENV_FOLDER=~/ift6759-project1-venv

cd $SLURM_TMPDIR/
cp -a $LOCAL_GIT_REPO_FOLDER/. $SLURM_TMPDIR/
cp -a $LOCAL_VENV_FOLDER/. $SLURM_TMPDIR/venv/
module load python/3.7.4
module load hdf5-mpi/1.10.3
source $SLURM_TMPDIR/venv/bin/activate

python trainer.py \
  --admin_cfg_path configs/admin/daily_daytime_01_train.json \
  --user_cfg_path configs/user/cnn_image_daily_daytime_v2_pretrained.json \
  --tensorboard_tracking_folder /project/cq-training-1/project1/teams/team03/tensorboard/$USER
