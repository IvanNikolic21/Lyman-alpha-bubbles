#!/bin/bash
#SBATCH --job-name=add_mock_x_v2
#SBATCH --array=0-9
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=add_mock_x_v2_%A_%a.out
#
# Simulated observations x for the v2 mocks (generate_bubble_mocks_v2.sh), for ONE intrinsic EW model given as the
# first argument:
#     sbatch add_mock_observations_v2.sh exponential
#     sbatch add_mock_observations_v2.sh tang_muv
#     sbatch add_mock_observations_v2.sh gagnon_hartman
# Output: bubble_mocks_v2_run1_x_<ew_model>/.
#
# 10 array tasks x 4 cores = 40 cores. Not 40 x 1: every task's galaxy draws already run in a joblib pool of 30
# worker processes (n_jobs=30, hard-coded in lyabubbles/speed_up.py get_content), so 40 tasks would mean 1200
# processes. 4 cores / 16G per task is the per-task size of the earlier add_mock_observations.sh run.
# Task i processes the batches with batch_idx % 10 == i (60 of the 600 batches). --time is a guess (each task
# does twice the earlier 60k-mock run): check the time between "[add_x:train] batch" lines in the first log and
# resubmit if needed -- finished batches are skipped. Keep --n_tasks equal to the array size when resubmitting.
#
# Start only after ALL generate_bubble_mocks_v2.sh tasks have finished (the script refuses otherwise, because a
# missing mock batch would shift the noise seeds). Either wait, or chain it:
#     sbatch --dependency=afterok:<generate_job_id> add_mock_observations_v2.sh exponential
#
# Needs the full py21cmfast-dependent stack, unlike generate_bubble_mocks.py.

EW_MODEL=${1:?usage: sbatch add_mock_observations_v2.sh <exponential|tang_muv|gagnon_hartman>}

python add_mock_observations.py \
    --mocks_dir bubble_mocks_v2_run1 \
    --output_dir "bubble_mocks_v2_run1_x_${EW_MODEL}" \
    --ew_model "${EW_MODEL}" \
    --seed 0 \
    --task_id "${SLURM_ARRAY_TASK_ID}" \
    --n_tasks 10
