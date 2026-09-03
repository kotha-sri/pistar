#!/bin/bash
set -e
export PYTHONNOUSERSITE=1
cd ~/rlpp
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rlpp

echo "=== Waiting for v3 training (PID $1) to finish ==="
while kill -0 "$1" 2>/dev/null; do
    sleep 60
done
echo "v3 training done."

echo ""
echo "=== Running ceiling sweep ==="
python -u eval_ceiling.py 2>&1 | tee eval_ceiling.log

echo ""
echo "=== Running raceline adaptation sweep ==="
python -u eval_raceline_adapt.py 2>&1 | tee eval_raceline_adapt.log

echo ""
echo "=== All evaluations complete ==="
