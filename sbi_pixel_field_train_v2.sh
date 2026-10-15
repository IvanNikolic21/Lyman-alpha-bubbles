#!/bin/bash
#SBATCH --job-name=train_nre_v2
#SBATCH --partition=astro_gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=train_nre_v2_%j.out
#SBATCH --error=train_nre_v2_%j.err
#
# Pixel-field NRE on the v2-prior mocks (1M training sims), for ONE intrinsic EW model given as the first argument:
#     sbatch sbi_pixel_field_train_v2.sh exponential
#     sbatch sbi_pixel_field_train_v2.sh tang_muv
#     sbatch sbi_pixel_field_train_v2.sh gagnon_hartman
# Same settings as sbi_pixel_field_train_gpu.sh (the bubble_mocks_run1_x / pixel_mock run), only the data and sizes
# differ.
#
# --mem: load_sims concatenates the float64 theta, 1M x 43 x 75 x 8 B = 26 GB, briefly twice (list + concatenated
# array), plus the float32 tensor (13 GB) -> ~65 GB peak; 128G leaves room. The H100 NVL has 94 GB, enough to hold
# the 13 GB float32 training set.
# --time: the 2-day partition cap. inference.train() has no checkpointing, so a run killed at the cap saves nothing;
# --max_num_epochs is the safety net. An epoch is ~20x longer than for the 50k-sim pixel_mock run (42 epochs).
# If that run's elapsed time x 20 x (epochs needed / 42) exceeds ~2 days, lower --max_num_epochs.

EW_MODEL=${1:?usage: sbatch sbi_pixel_field_train_v2.sh <exponential|tang_muv|gagnon_hartman>}

source ~/miniconda3/etc/profile.d/conda.sh
conda activate UVLF_clust

python /groups/astro/ivannik/programs/Lyman-alpha-bubbles/sbi_pixel_field.py train_nre \
    --sims_dir "bubble_mocks_v2_run1_x_${EW_MODEL}" \
    --output_dir "sbi_runs/pixel_v2prior_${EW_MODEL}" \
    --n_los 75 \
    --embedding cnn \
    --device cuda \
    --max_num_epochs 100
