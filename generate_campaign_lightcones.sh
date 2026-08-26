#!/bin/bash
#SBATCH --job-name=lya_campaign
#SBATCH --array=0-39%10
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/lustre/astro/ivannik/21cmFAST_cache/lightcone_campaign_logs/%A_%a.out
#
# One array task per (F_ESC10, seed) combination -- 5 F_ESC10 values x 8 seeds
# = 40 tasks (array=0-39; if you change N_SEEDS in the .py, update this range
# to match: N_ESC10 * N_SEEDS - 1).
#
# JOB SPECS -- deliberately generous, not measured (see cluster-storage-
# expansion memory): --cpus-per-task=16 is solid, this exact N_THREADS
# already ran successfully in every calibration run this session. --mem=32G
# and --time=6h are safe-but-ungrounded guesses -- after the first few array
# tasks complete, check `sacct -j <jobid> --format=JobID,Elapsed,MaxRSS` and
# report back so these can be right-sized for a resubmission (smaller specs
# queue faster and are more considerate of shared cluster resources).
#
# %10 throttles to at most 10 concurrent tasks (160 cores) rather than all
# 40 at once (640 cores) -- a considerate default given no data on this
# cluster's fair-share/queue policy; raise or drop the %10 if your
# allocation comfortably supports more concurrency.
#
# Safe to submit the whole array in one `sbatch` call (unlike
# calibrate_fesc_to_tau.py's sequential single-task design) -- each task is
# independent and idempotent via its own marker file, so there's no shared
# state to race on. A task that's already done (marker exists) just prints
# "[skip]" and exits immediately, so relaunching the same array after a
# partial run/failure is also safe.
#
# CHECK BEFORE SUBMITTING: --cpus-per-task above MUST match N_THREADS in
# generate_campaign_lightcones.py's BASE_OVERRIDES (currently 16) -- and
# double-check this is the REAL #SBATCH line, not silently commented out
# with an extra '#' (see cluster-workflow-notes memory).

mkdir -p /lustre/astro/ivannik/21cmFAST_cache/lightcone_campaign_logs/

python generate_campaign_lightcones.py "${SLURM_ARRAY_TASK_ID}"
