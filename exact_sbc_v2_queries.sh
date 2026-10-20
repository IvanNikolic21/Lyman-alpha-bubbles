#!/bin/bash
#SBATCH --job-name=exact_sbc_queries
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=exact_sbc_queries_%j.out
#
# Stage 1 of exact_pixel_sbc.py (SBC of the exact-likelihood posterior on the v2 pool), for ONE intrinsic EW model
# given as the first argument:
#     sbatch exact_sbc_v2_queries.sh exponential      (likewise tang_muv, gagnon_hartman)
# Picks N_QUERIES (default 200) query mocks from the val split (same rows for every EW model: --seed 0) and saves
# their x, true maps and true hyperparameters to sbc_exact_v2_<ew_model>/sbc_queries.npz. Light: minutes.

EW_MODEL=${1:?usage: sbatch exact_sbc_v2_queries.sh <exponential|tang_muv|gagnon_hartman>}
REPO=${REPO:-/groups/astro/ivannik/programs/Lyman-alpha-bubbles}
N_QUERIES=${N_QUERIES:-200}
export OMP_NUM_THREADS=1

python "${REPO}/exact_pixel_sbc.py" queries \
    --pool_dir "bubble_mocks_v2_run1_x_${EW_MODEL}" \
    --mocks_dir bubble_mocks_v2_run1 \
    --n_queries "${N_QUERIES}" \
    --output_dir "sbc_exact_v2_${EW_MODEL}" \
    --lya_catalog "${REPO}/tb_lya.txt" \
    --properties_catalog "${REPO}/sample_nirspec_properties.txt"
