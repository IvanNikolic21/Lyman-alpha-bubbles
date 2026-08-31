#!/bin/bash
#SBATCH --job-name=bubble_mocks
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=bubble_mocks_%j.out
#SBATCH --error=bubble_mocks_%j.err
#
# Pure geometry (no py21cmfast, no simulation) -- single-threaded is fine,
# this is nothing like the 21cmFAST lightcone jobs. The first production run
# (15000+3000 mocks) completed comfortably inside the old 2-hour limit, so
# 4 hours is generous headroom for this 50000+10000 scale-up (~3.3x the
# work), not a measured number.
#
# Resumable: safe to resubmit this exact script if it gets killed partway
# through -- already-written batch files are skipped, not regenerated. Also
# safe to re-run with a LARGER --n_sim/--n_sim_val in the same --output_dir
# to extend an existing dataset (this is exactly what this run now does,
# scaling the first production run's 15000+3000 up to the 50000+10000 used
# elsewhere in this pipeline's real pixel-field SBI training) -- verified
# locally that extending reproduces byte-identical results to a from-scratch
# run at the larger size, after fixing a real bug where skipped batches used
# to leave the master RNG un-advanced (see generate_bubble_mocks.py's
# generate_split docstring/comments) and would otherwise have silently
# duplicated early batches' mocks into the new ones.
# Needs numpy/scipy/pandas/astropy (real_data.py's actual dependencies,
# NOT py21cmfast) in whatever env this runs under.

python generate_bubble_mocks.py \
    --n_sim 50000 \
    --n_sim_val 10000 \
    --batch_size 2000 \
    --output_dir bubble_mocks_run1 \
    --seed 0
