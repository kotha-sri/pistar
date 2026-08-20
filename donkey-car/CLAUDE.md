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

Replicating RLPP paper (arXiv 2501.17311v2) — residual RL + Pure Pursuit on F1TENTH.
See `scripts/f1tenth/plan.md` for the full roadmap.
