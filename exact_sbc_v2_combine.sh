#!/bin/bash
#SBATCH --job-name=exact_sbc_combine
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=exact_sbc_combine_%j.out
#
# Stage 3 of exact_pixel_sbc.py: posterior weights per query, hyperparameter ranks, map statistics, TARP, depth
# sweep, for ONE intrinsic EW model (first argument). Run after exact_sbc_v2_loglik.sh:
#     sbatch --dependency=afterok:<loglik_job_id> exact_sbc_v2_combine.sh exponential
# Output: sbc_exact_v2_<ew_model>/exact_sbc.npz (same keys as sbi_pixel_sbc.py's pixel_sbc.npz + hyper_*), so
# plot_pixel_sbc.py / plot_pixel_sbc_depth.py read it unchanged.
# Memory: weights 200 queries x 1.2M (1.9 GB float64) + the float32 log-likelihoods; theta is streamed.

EW_MODEL=${1:?usage: sbatch exact_sbc_v2_combine.sh <exponential|tang_muv|gagnon_hartman>}
REPO=${REPO:-/groups/astro/ivannik/programs/Lyman-alpha-bubbles}
export OMP_NUM_THREADS=1

python "${REPO}/exact_pixel_sbc.py" combine \
    --pool_dir "bubble_mocks_v2_run1_x_${EW_MODEL}" \
    --mocks_dir bubble_mocks_v2_run1 \
    --output_dir "sbc_exact_v2_${EW_MODEL}" \
    --lya_catalog "${REPO}/tb_lya.txt" \
    --properties_catalog "${REPO}/sample_nirspec_properties.txt"
