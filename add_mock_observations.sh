#!/bin/bash
#SBATCH --job-name=add_mock_x
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=add_mock_x_%j.out
#SBATCH --error=add_mock_x_%j.err
#
# Needs the FULL py21cmfast-dependent stack (real_data_run.py ->
# speed_up.py -> galaxy_prop.py), unlike generate_bubble_mocks.py itself --
# run this in whatever env/on whatever machine sbi_pixel_field.py's own
# `simulate` command runs in.
#
# No forked worker pool here (single process, unlike sbi_real_data.py/
# sbi_pixel_field.py's own `simulate`), so BLAS threading is left alone
# rather than capped to 1 -- --cpus-per-task=4 gives it room. Should be
# noticeably CHEAPER than sbi_pixel_field.py's own `simulate` at the same
# n_sim: this script skips the expensive multi-box 21cmFAST ray-trace
# entirely (theta is already on disk from generate_bubble_mocks.py) and
# only pays for _refresh_mc_state + segments_to_tau + noise injection.
# --time/--mem are still an UNTESTED guess for the first real run at this
# dataset's scale (60000 total sims x 43 galaxies) -- tighten after
# watching it.
#
# Resumable: safe to resubmit, and safe to re-run after
# generate_bubble_mocks.py is extended to a larger --n_sim -- only
# processes whatever {prefix}_batch_*.npz files exist in --mocks_dir,
# skipping any already-built output batch.

python add_mock_observations.py \
    --mocks_dir bubble_mocks_run1 \
    --output_dir bubble_mocks_run1_x \
    --ew_model exponential \
    --seed 0