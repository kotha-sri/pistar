"""Evaluate a saved fault-Reptile meta checkpoint across the held-out fault set.

Loads a meta_params_*.pkl and reports few-lap recovery (completions at several
adapt budgets) on the in-distribution (trained fault types, held-out severity)
and out-of-distribution (never-trained fault types) tasks. Standalone so it can
be chained to run automatically on pistar after a training run finishes.

  python eval_faults_ckpt.py --ckpt reptile/reptile_faults_v2/meta_params_best.pkl \
      --track Austin --adapt 0 2000 4000 10000
"""

import argparse
import json
import os
import types

from train_reptile import load_meta_checkpoint
from train_reptile_faults import evaluate_fault_adaptation, base_env_kwargs
import fault_distribution as fd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to meta_params_*.pkl (abs or rel to this dir)")
    ap.add_argument("--track", default="Austin")
    ap.add_argument("--adapt", type=int, nargs="+", default=[0, 2000, 4000, 10000])
    ap.add_argument("--out", default="reptile_faults_v2_eval.json")
    a = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    tracks_dir = os.path.join(script_dir, "f1tenth_racetracks")
    ckpt = a.ckpt if os.path.isabs(a.ckpt) else os.path.join(script_dir, a.ckpt)

    args = types.SimpleNamespace(
        inner_lr=5e-4, inner_steps=max(a.adapt), batch_size=256, utd=20, seed=42,
        action_scaling=[0.05, 1.0], pp_reference="centerline", pp_blend=1.0, smoke=False)
    env_kwargs = base_env_kwargs(args)

    meta = load_meta_checkpoint(ckpt)
    id_tasks = [(a.track, n, 0.5) for n in fd.TRAIN_FAULTS]
    ood_tasks = [(a.track, n, 0.4) for n in fd.HELD_OUT_FAULTS]

    print(f"=== eval {os.path.basename(ckpt)} on {a.track}, adapt {a.adapt} ===")
    print(f"    (completions per adapt budget; 0-adapt = zero-shot robustness)")
    out = {"ckpt": ckpt, "track": a.track, "adapt": a.adapt}
    for label, tasks in (("ID", id_tasks), ("OOD", ood_tasks)):
        res = evaluate_fault_adaptation(meta, tasks, tracks_dir, args, a.adapt, env_kwargs)
        out[label] = res
        for key, cells in res.items():
            summ = "  ".join(f"{s}:{cells[s]['completed']}/3" for s in a.adapt)
            print(f"  [{label}] {key:<22s} {summ}")
    with open(os.path.join(script_dir, a.out), "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
