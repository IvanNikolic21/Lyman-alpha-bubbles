#!/bin/bash
#SBATCH --job-name=bubble_mocks_v2
#SBATCH --array=0-39
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=bubble_mocks_v2_%A_%a.out
#
# v2 prior (bubble_prior_v2.py): 1M training + 200k validation mocks, split over 40 single-core array tasks.
# Task i computes the batches with batch_idx % 40 == i (500 train + 100 val batches of 2000 mocks, so 12-13 train
# and 2-3 val batches per task). The mocks do not depend on how many tasks there are (checked: a 3-task run is
# byte-identical to a 1-task run). Measured locally at ~25 mocks/s/core, i.e. ~80 s per batch, ~25 min per task;
# --time is ~10x headroom for a slower cluster core.
#
# Resumable: resubmitting the same array skips every batch file that already exists. Keep --n_tasks equal to the
# array size, and --batch_size/--seed unchanged, when resubmitting into the same --output_dir.
#
# bsd_lu24_calibration.npz (output of calibrate_bsd_lu24.py) must be on the cluster at --bsd_calibration.
# Output size: ~3.9 GB of uint8 masks (1.2M x 43 x 75).

export OMP_NUM_THREADS=1      # one core per task: keep numpy/scipy from starting extra threads

python generate_bubble_mocks.py \
    --prior v2 \
    --bsd_calibration bsd_lu24_calibration.npz \
    --n_sim 1000000 \
    --n_sim_val 200000 \
    --batch_size 2000 \
    --output_dir bubble_mocks_v2_run1 \
    --seed 0 \
    --task_id "${SLURM_ARRAY_TASK_ID}" \
    --n_tasks 40
