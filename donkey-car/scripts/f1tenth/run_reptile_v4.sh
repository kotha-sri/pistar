#!/bin/bash
set -e
export PYTHONNOUSERSITE=1
cd ~/rlpp
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rlpp

python -u train_reptile.py \
    --meta-iterations 200 \
    --inner-steps 20000 \
    --inner-envs 4 \
    --utd 20 \
    --inner-lr 1e-3 \
    --batch-size 256 \
    --epsilon-start 0.3 \
    --epsilon-end 0.05 \
    --warmup-steps 0 \
    --include-generated \
    --eval-every 25 \
    --save-every 25 \
    --tag reptile_v4 \
    --seed 42 \
    --resume reptile/reptile_v3/meta_params_best.pkl \
    --reward-alpha-raceline 0.5 \
    --reward-use-raceline-progress
