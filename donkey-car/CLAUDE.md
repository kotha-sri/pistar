# Project: Autonomous Racing RL

## Remote Compute

Training runs on a remote GPU machine via SSH:
- **Host:** `pistar@10.28.177.234`
- **SSH key:** `/c/Users/kotha/.ssh/id_pistar` (use forward slashes in Bash tool)
- **GPU:** NVIDIA RTX 4060 (8GB)
- **Conda env:** `rlpp` (Python 3.10)
- **Code location:** `~/rlpp/`
- **Must set** `export PYTHONNOUSERSITE=1` before running anything (old system-level tensorflow/protobuf conflicts otherwise)

Run training via wrapper script:
```bash
nohup ~/rlpp/run_train.sh > ~/rlpp/train.log 2>&1 &
```

## Local Development

- Working directory: `scripts/f1tenth/`
- Vendored F1TENTH sim at `scripts/f1tenth/rpl4f110/simulator/`
- Track data at `scripts/f1tenth/f1tenth_racetracks/`
- Local `.venv` at repo root for testing

## Current Work

**Pivoted (2026-09-08) to resilient fault adaptation.** The RLPP replication (residual RL + Pure
Pursuit on F1TENTH, arXiv 2501.17311v2) is now the reusable groundwork, not the goal. New north
star: online, self-supervised, safe adaptation to **out-of-model faults** (steering bias, actuator
loss-of-effectiveness/latency, low-grip patch, etc.) that classical adaptive control / sysID cannot
handle. Racing/F1TENTH is the fault-injectable testbed.

- **Active charter:** `scripts/f1tenth/fault_adaptation_plan.md` (supersedes `plan.md`, kept for history).
- **Phases 0–D landed** (see `scripts/f1tenth/fault_adaptation_plan.md`): Phase 0 fault injection;
  A/B `eval_fault_matrix.py` (in-model vs out-of-model crossover substrate); C `train_reptile_faults.py`
  (Reptile-over-faults — v1 run showed robust zero-shot generalization to unseen fault *types*, plus
  meta-instability; `reptile_faults_v1_results.md`); D `fault_detector.py` (position-conditioned
  detector + fallback guardrail, smoke-passing).
- **Next:** launch `reptile_faults_v2` on pistar (stability fixes: `--meta-clip 5.0 --inner-lr 5e-4
  --epsilon-end 0.05`) when the box is reachable; then Phase E (integrate detector+guardrail+online
  adaptation for mid-run recovery) and port MAP/Follow-the-Gap as extra standard baselines.
