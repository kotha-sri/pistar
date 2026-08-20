#!/bin/bash
set -e
export PYTHONNOUSERSITE=1
cd ~/rlpp
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rlpp

python -u train_reptile.py \
    --meta-iterations 400 \
    --inner-steps 20000 \
    --inner-envs 4 \
    --utd 20 \
    --inner-lr 1e-3 \
    --batch-size 256 \
    --epsilon-start 1.0 \
    --epsilon-end 0.1 \
    --warmup-steps 50000 \
    --include-generated \
    --eval-every 25 \
    --save-every 25 \
    --tag reptile_v3 \
    --seed 42
