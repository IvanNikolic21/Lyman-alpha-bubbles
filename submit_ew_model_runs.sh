#!/bin/bash
# Re-run the mock-trained pixel-field SBI with a different intrinsic Lya EW model.
# theta (the bubble mocks) is unchanged; only x is regenerated, so each model gets
#   1. add_x  : bubble_mocks_run1 -> bubble_mocks_run1_x_<model>   (CPU, needs py21cmfast stack)
#   2. train  : NRE on those pairs -> sbi_runs/pixel_mock_<model>   (GPU)
#   3. infer  : real-data posterior -> sbi_runs/pixel_mock_<model>/pixel_infer.npz
#   4. sbc    : SBC + depth sweep  -> sbi_runs/pixel_mock_<model>/pixel_sbc.npz
# chained with afterok dependencies (3 and 4 run in parallel after 2).
#
# Usage (from the directory holding bubble_mocks_run1/, sbi_runs/, tb_lya.txt -- the same
# place sbi_pixel_field_train_gpu.sh was submitted from):
#   bash submit_ew_model_runs.sh tang_muv gagnon_hartman
#
# Why tang_muv and not `tang`: the Tang+24 z=6.5-8 table is the OBSERVED EW distribution at our
# redshifts, i.e. already IGM-attenuated; using it as intrinsic would double-count the IGM.
# tang_muv is the z~5-6 Muv-dependent fit (arXiv:2402.06070), the conventional intrinsic reference.
#
# Resources copy the exponential run (add_mock_observations.sh / sbi_pixel_field_train_gpu.sh).
set -euo pipefail

REPO=/groups/astro/ivannik/programs/Lyman-alpha-bubbles
ENV='source ~/miniconda3/etc/profile.d/conda.sh && conda activate UVLF_clust'
MOCKS=bubble_mocks_run1

[ $# -ge 1 ] || { echo "usage: bash $0 <ew_model> [<ew_model> ...]"; exit 1; }
[ -d "$MOCKS" ] || { echo "no $MOCKS/ here -- submit from the directory that holds it"; exit 1; }
[ -f tb_lya.txt ] || { echo "no tb_lya.txt here -- submit from the run directory"; exit 1; }
mkdir -p logs

for M in "$@"; do
    X=${MOCKS}_x_${M}
    RUN=sbi_runs/pixel_mock_${M}

    j1=$(sbatch --parsable --job-name=addx_${M} --ntasks=1 --cpus-per-task=4 --mem=16G \
        --time=03:00:00 --output=logs/addx_${M}_%j.out --error=logs/addx_${M}_%j.err \
        --wrap="$ENV && python $REPO/add_mock_observations.py --mocks_dir $MOCKS \
                 --output_dir $X --ew_model $M --seed 0")

    j2=$(sbatch --parsable --dependency=afterok:$j1 --job-name=nre_${M} \
        --partition=astro_gpu --gres=gpu:nvidia_h100_nvl:1 --ntasks=1 --cpus-per-task=8 \
        --mem=32G --time=04:00:00 --output=logs/nre_${M}_%j.out --error=logs/nre_${M}_%j.err \
        --wrap="$ENV && python $REPO/sbi_pixel_field.py train_nre --sims_dir $X \
                 --output_dir $RUN --n_los 75 --embedding cnn --device cuda --max_num_epochs 200")

    j3=$(sbatch --parsable --dependency=afterok:$j2 --job-name=inf_${M} --ntasks=1 \
        --cpus-per-task=4 --mem=16G --time=01:00:00 \
        --output=logs/inf_${M}_%j.out --error=logs/inf_${M}_%j.err \
        --wrap="$ENV && python $REPO/sbi_pixel_field.py infer --n_los 75 \
                 --ratio $RUN/ratio_estimator.pt --pool_dir $X --pool_split val \
                 --n_samples 1000 --device cpu --output_dir $RUN")

    j4=$(sbatch --parsable --dependency=afterok:$j2 --job-name=sbc_${M} --ntasks=1 \
        --cpus-per-task=8 --mem=32G --time=08:00:00 \
        --output=logs/sbc_${M}_%j.out --error=logs/sbc_${M}_%j.err \
        --wrap="$ENV && python $REPO/sbi_pixel_sbc.py --n_los 75 \
                 --ratio $RUN/ratio_estimator.pt --pool_dir $X --pool_split val \
                 --n_queries 1000 --n_post 200 --device cpu --output_dir $RUN")

    echo "$M: add_x $j1 -> train $j2 -> infer $j3 + sbc $j4   (logs/, output $RUN)"
done
