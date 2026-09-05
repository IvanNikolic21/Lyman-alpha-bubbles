#!/bin/bash
#SBATCH --job-name=train_nre_gpu
#SBATCH --partition=astro_gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=train_nre_gpu_%j.out
#SBATCH --error=train_nre_gpu_%j.err
#
# Resubmit of the CPU run that got SLURM-cancelled at the astro2_short
# 10h/12h wall-clock cap mid-training (inference.train() has no
# checkpointing -- see sbi_pixel_field.py's run_train_nre docstring/the
# --max_num_epochs flag added after that). Moving to astro_gpu (H100,
# 2-day cap) for two reasons: a ResNet-style classifier (--embedding cnn)
# is built for GPU throughput and was plausibly the real cost driver on
# CPU regardless of core count, and the 2-day cap gives enormous headroom
# over the previous 10h even if GPU doesn't speed things up as much as
# hoped.
#
# --time=4h is a GUESS for the first real GPU run at this dataset size
# (50k sims) -- genuinely unmeasured, but should have a lot of headroom if
# GPU acceleration works as expected; --max_num_epochs is set as a second,
# independent safety net so a slower-than-expected run still ends
# predictably instead of getting SLURM-killed with zero saved output again.
# --mem=32G / --cpus-per-task=8 are far less than the original failed
# job's 100G/16 cpus -- that request looks leftover from a different,
# copy-pasted job template (a 21cmFAST lightcone-generation array job),
# not actually sized for this job, which is GPU/single-process, not a
# CPU-heavy array job. Actual sims_dir data (50k+10k sims, theta+x) is
# well under 1GB.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate UVLF_clust

python /groups/astro/ivannik/programs/Lyman-alpha-bubbles/sbi_pixel_field.py train_nre \
    --sims_dir bubble_mocks_run1_x \
    --output_dir sbi_runs/pixel_mock \
    --n_los 75 \
    --embedding cnn \
    --device cuda \
    --max_num_epochs 200