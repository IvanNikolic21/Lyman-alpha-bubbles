#!/bin/bash
#SBATCH --job-name=bubble_mocks
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=bubble_mocks_%j.out
#SBATCH --error=bubble_mocks_%j.err
#
# Pure geometry (no py21cmfast, no simulation) -- single-threaded is fine,
# this is nothing like the 21cmFAST lightcone jobs. --time is a generous
# guess for a real production size (default --n_sim 15000); the local test
# (144 mocks) took well under 2 minutes after the exponnorm.ppf hang fix
# (see generate_bubble_mocks.py's draw_radii docstring), so this should
# have a lot of headroom, but genuinely unmeasured at 15000+ scale --
# tighten after watching the first real run.
#
# Resumable: safe to resubmit this exact script if it gets killed partway
# through -- already-written batch files are skipped, not regenerated.
# Needs numpy/scipy/pandas/astropy (real_data.py's actual dependencies,
# NOT py21cmfast) in whatever env this runs under.

python generate_bubble_mocks.py \
    --n_sim 15000 \
    --batch_size 2000 \
    --output_dir bubble_mocks_run1 \
    --seed 0
