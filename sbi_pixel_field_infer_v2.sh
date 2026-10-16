#!/bin/bash
#SBATCH --job-name=infer_nre_v2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=infer_nre_v2_%j.out
#
# Network posterior at the real data on the v2 val pool (200k maps, NOT used in training), for ONE intrinsic EW
# model given as the first argument:
#     sbatch sbi_pixel_field_infer_v2.sh exponential
# CPU only: a forward pass, no training (no GPU needed). Output: sbi_runs/pixel_v2prior_<ew>/pixel_infer.npz.
# Compare with the exact likelihood on the SAME pool (SPLITS=val sbatch exact_v2_combine.sh <ew>) using
# compare_exact_nn_v2.py.
#
# Memory: the val theta as float64 (200k x 3225 x 8 B = 5.2 GB) + its (n_samples) joint resamples; the forward pass
# runs in chunks of 5000 maps. --time is a guess.

EW_MODEL=${1:?usage: sbatch sbi_pixel_field_infer_v2.sh <exponential|tang_muv|gagnon_hartman>}
REPO=${REPO:-/groups/astro/ivannik/programs/Lyman-alpha-bubbles}
export OMP_NUM_THREADS=8

source ~/miniconda3/etc/profile.d/conda.sh
conda activate UVLF_clust

python "${REPO}/sbi_pixel_field.py" infer \
    --ratio "sbi_runs/pixel_v2prior_${EW_MODEL}/ratio_estimator.pt" \
    --pool_dir "bubble_mocks_v2_run1_x_${EW_MODEL}" \
    --pool_split val \
    --lya_catalog "${REPO}/tb_lya.txt" \
    --properties_catalog "${REPO}/sample_nirspec_properties.txt" \
    --main_dir "${REPO}/" \
    --device cpu \
    --output_dir "sbi_runs/pixel_v2prior_${EW_MODEL}"
