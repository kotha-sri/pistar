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
- **Phase 0 DONE:** `fault_injection.py` + `residual_env.py` (`faults=` kwarg) + `demo_fault_injection.py`
  (`python demo_fault_injection.py --track Spielberg`).
- Next: Phase A/B — fault taxonomy build-out + `eval_fault_matrix.py` (the in-model vs out-of-model crossover).
