#!/bin/bash
#SBATCH --job-name=exact_v2_loglik
#SBATCH --array=0-39
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=exact_v2_loglik_%A_%a.out
#
# Stage 1 of exact_pixel_likelihood_chunked.py on the v2 pool (train + val = 1.2M maps), for ONE intrinsic EW model
# given as the first argument:
#     sbatch exact_v2_loglik.sh exponential      (likewise tang_muv, gagnon_hartman)
# Output: exact_v2_<ew_model>/loglik_<split>_batch_XXXXX.npz, one per pool batch file.
#
# 40 single-core tasks; the 600 pool files are dealt round-robin (15 per task). Measured locally: ~10 s per
# 2000-map batch, i.e. ~3 min per task; --time has wide headroom. Resumable: resubmit to fill in missing files.
# No py21cmfast needed (numpy/scipy/astropy/pandas/h5py), the same env as generate_bubble_mocks.py.

EW_MODEL=${1:?usage: sbatch exact_v2_loglik.sh <exponential|tang_muv|gagnon_hartman>}
REPO=${REPO:-/groups/astro/ivannik/programs/Lyman-alpha-bubbles}
export OMP_NUM_THREADS=1      # one core per task

python "${REPO}/exact_pixel_likelihood_chunked.py" loglik \
    --ew_model "${EW_MODEL}" \
    --pool_dir "bubble_mocks_v2_run1_x_${EW_MODEL}" \
    --splits train val \
    --output_dir "exact_v2_${EW_MODEL}" \
    --lya_catalog "${REPO}/tb_lya.txt" \
    --properties_catalog "${REPO}/sample_nirspec_properties.txt" \
    --task_id "${SLURM_ARRAY_TASK_ID}" \
    --n_tasks 40
