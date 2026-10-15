#!/bin/bash
#SBATCH --job-name=exact_v2_combine
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=exact_v2_combine_%j.out
#
# Stage 2 of exact_pixel_likelihood_chunked.py: posterior weights, full and leave-one-out maps, and the v2 prior
# parameters of every pool map, for ONE intrinsic EW model (first argument). Run after exact_v2_loglik.sh:
#     sbatch --dependency=afterok:<loglik_job_id> exact_v2_combine.sh exponential
# Output: exact_v2_<ew_model>.npz.
#
# Optional second argument "tau": also apply a Gaussian tau_e prior, Planck 2018 (0.054 +- 0.007) with the sd
# doubled, as an importance weight -> exact_v2_<ew_model>_tau.npz. Caveat: for low x_HI(7) with late z_end the tanh
# history is very broad and tau_e is set by its extrapolation to z >> 7.3, outside our sightlines.
#
# Memory: per-galaxy logliks 1.2M x 43 (0.4 GB) + weights of 47 sets: full + 43 single galaxies + 3 groups (0.5 GB); theta is streamed file by file.
# Time: dominated by reading the 600 pool files once (~31 GB of float64 theta) -- --time is a guess.

EW_MODEL=${1:?usage: sbatch exact_v2_combine.sh <exponential|tang_muv|gagnon_hartman> [tau]}
REPO=${REPO:-/groups/astro/ivannik/programs/Lyman-alpha-bubbles}
TAU_ARGS=()
SUFFIX=""
if [ "${2:-}" = "tau" ]; then
    TAU_ARGS=(--tau_prior 0.054 0.014)
    SUFFIX="_tau"
fi

python "${REPO}/exact_pixel_likelihood_chunked.py" combine \
    --ew_model "${EW_MODEL}" \
    --pool_dir "bubble_mocks_v2_run1_x_${EW_MODEL}" \
    --splits train val \
    --loglik_dir "exact_v2_${EW_MODEL}" \
    --mocks_dir bubble_mocks_v2_run1 \
    --lya_catalog "${REPO}/tb_lya.txt" \
    --properties_catalog "${REPO}/sample_nirspec_properties.txt" \
    --groups 14-18 27-36 21,22,24,25 \
    "${TAU_ARGS[@]}" \
    --output "exact_v2_${EW_MODEL}${SUFFIX}.npz"
