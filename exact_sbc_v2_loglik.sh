#!/bin/bash
#SBATCH --job-name=exact_sbc_loglik
#SBATCH --array=0-39
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --output=exact_sbc_loglik_%A_%a.out
#
# Stage 2 of exact_pixel_sbc.py: log-likelihood of every v2 pool map (train + val, 1.2M) for every query, for ONE
# intrinsic EW model (first argument). Run after exact_sbc_v2_queries.sh:
#     sbatch --dependency=afterok:<queries_job_id> exact_sbc_v2_loglik.sh exponential
# Output: sbc_exact_v2_<ew_model>/sbcll_<split>_batch_XXXXX.npz, one per pool file.
#
# Cost (measured locally): ~0.5 s per detected galaxy per 2000-map file; mocks have ~5.6 detections each, so
# 200 queries -> ~10 min per file, 15 files per task -> ~2.5-4 h per task. Resumable: resubmit to fill in
# missing files. Plain python env (numpy/scipy/astropy/pandas), as exact_v2_loglik.sh.

EW_MODEL=${1:?usage: sbatch exact_sbc_v2_loglik.sh <exponential|tang_muv|gagnon_hartman>}
REPO=${REPO:-/groups/astro/ivannik/programs/Lyman-alpha-bubbles}
export OMP_NUM_THREADS=1      # one core per task

python "${REPO}/exact_pixel_sbc.py" loglik \
    --ew_model "${EW_MODEL}" \
    --pool_dir "bubble_mocks_v2_run1_x_${EW_MODEL}" \
    --output_dir "sbc_exact_v2_${EW_MODEL}" \
    --lya_catalog "${REPO}/tb_lya.txt" \
    --properties_catalog "${REPO}/sample_nirspec_properties.txt" \
    --task_id "${SLURM_ARRAY_TASK_ID}" \
    --n_tasks 40
