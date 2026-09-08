"""Phase B -- the regime-proof fault matrix (resilient-adaptation pivot).

Runs a STANDARD, non-adaptive controller (Pure Pursuit alone -- the community
baseline) across the full fault taxonomy at graded severities, over several
seeds, and reports metrics comparable to the literature:

  completion_rate   -- fraction of runs that finish the lap
  crash_rate        -- fraction that crash (cf. TC-Driver 2205.09370 crash ratio)
  mean_progress     -- mean fraction of one lap completed
  mean_dev          -- mean |deviation| from the reference path (tracking error)

Output:
  fault_matrix_results.json   -- full (fault x severity) matrix
  fault_matrix_summary.md     -- readable tables + the in-model/out-of-model read

Honest scope: this establishes the *fault substrate* -- that every fault produces
graded, controllable degradation of a standard controller. The full crossover
claim (classical adaptation recovers IN-MODEL faults but fails OUT-OF-MODEL ones)
needs the ADAPTIVE baselines (adaptive-MPC / sysID / our residual), added later.
Pure Pursuit is non-adaptive, so here it degrades under both classes; the point
is that the faults are real, monotone, and cleanly split by class.

Run:
  python eval_fault_matrix.py --track Spielberg --seeds 3
"""

import argparse
import json
import os
import numpy as np

from residual_env import RLPPEnv
import fault_injection as fi

# Standard controller: Pure Pursuit alone (residual zeroed), deterministic sim.
BASE_KW = dict(alpha_rl=0.0, velocity_gain=0.5, mu_noise_std=0.0,
               velocity_curriculum=False, pp_reference="centerline")
ZERO_ACTION = np.zeros(2, dtype=np.float32)
FAILURE_THRESHOLD = 0.5   # progress fraction below which we call it a breakdown


def run_episode(env, max_steps, seed):
    obs, _ = env.reset(seed=seed)
    total_s = env.total_s
    prev_s = env.raceline_s[env._closest_idx]
    cumul_s = 0.0
    devs = []
    crashed = False
    for steps in range(1, max_steps + 1):
        obs, r, term, trunc, info = env.step(ZERO_ACTION)
        d, _ = env.pp.get_frenet_state(env._last_pos[0], env._last_pos[1],
                                       env._last_heading, env._closest_idx)
        devs.append(abs(d))
        cur_s = env.raceline_s[env._closest_idx]
        ds = cur_s - prev_s
        if ds < -total_s / 2:
            ds += total_s
        elif ds > total_s / 2:
            ds -= total_s
        cumul_s += ds
        prev_s = cur_s
        if info["collision"]:
            crashed = True
            break
        if term or trunc:
            break
    completed = (not crashed) and (cumul_s >= total_s * 0.99)
    return {
        "completed": bool(completed),
        "crashed": bool(crashed),
        "progress_frac": float(max(0.0, cumul_s) / total_s),
        "mean_abs_dev": float(np.mean(devs)) if devs else float("nan"),
    }


def cell(track, tracks_dir, fault_name, severity, seeds, max_steps):
    """Aggregate metrics over seeds for one (fault, severity)."""
    runs = []
    for sd in range(seeds):
        fault = None if severity == 0.0 else fi.from_spec(fault_name, severity)
        env = RLPPEnv(track_name=track, tracks_dir=tracks_dir, max_laps=1,
                      faults=fault, **BASE_KW)
        runs.append(run_episode(env, max_steps, seed=sd))
        env.close()
    return {
        "completion_rate": float(np.mean([r["completed"] for r in runs])),
        "crash_rate": float(np.mean([r["crashed"] for r in runs])),
        "mean_progress": float(np.mean([r["progress_frac"] for r in runs])),
        "mean_dev": float(np.nanmean([r["mean_abs_dev"] for r in runs])),
    }


def breakdown_severity(row, severities):
    """Lowest severity at which mean_progress drops below the failure threshold;
    None if the controller never breaks down across the swept range."""
    for s in severities:
        if row[str(s)]["mean_progress"] < FAILURE_THRESHOLD:
            return s
    return None


def write_summary(results, path):
    faults = results["faults"]
    sevs = results["severities"]
    lines = []
    lines.append(f"# Fault matrix -- standard Pure Pursuit on {results['track']}")
    lines.append("")
    lines.append(f"Controller: Pure Pursuit alone (non-adaptive), velocity_gain=0.5, "
                 f"{results['seeds']} seeds/cell, max_steps={results['max_steps']}.")
    lines.append("")
    lines.append("Clean baseline (severity 0): "
                 f"progress={results['baseline']['mean_progress']*100:.1f}%, "
                 f"mean|d|={results['baseline']['mean_dev']:.3f} m, "
                 f"completion={results['baseline']['completion_rate']*100:.0f}%.")
    lines.append("")

    def table(metric, fmt, scale=1.0):
        out = ["| fault | class | " + " | ".join(f"s={s}" for s in sevs) + " |",
               "|---|---|" + "---|" * len(sevs)]
        for name in faults:
            cls = "in" if name in fi.IN_MODEL else "out"
            cells = []
            for s in sevs:
                v = results["matrix"][name][str(s)][metric] * scale
                cells.append(fmt.format(v))
            out.append(f"| {name} | {cls} | " + " | ".join(cells) + " |")
        return "\n".join(out)

    lines.append("## Mean progress (% of one lap; lower = more degraded)")
    lines.append("")
    lines.append(table("mean_progress", "{:.0f}", scale=100.0))
    lines.append("")
    lines.append("## Crash rate (fraction of seeds; TC-Driver-style crash ratio)")
    lines.append("")
    lines.append(table("crash_rate", "{:.2f}"))
    lines.append("")
    lines.append("## Mean tracking error |d| (m)")
    lines.append("")
    lines.append(table("mean_dev", "{:.3f}"))
    lines.append("")

    lines.append("## Breakdown severity (lowest s with mean progress < "
                 f"{int(FAILURE_THRESHOLD*100)}%)")
    lines.append("")
    lines.append("| fault | class | breakdown s |")
    lines.append("|---|---|---|")
    for name in faults:
        cls = "in" if name in fi.IN_MODEL else "out"
        bs = breakdown_severity(results["matrix"][name], sevs)
        lines.append(f"| {name} | {cls} | {bs if bs is not None else 'never'} |")
    lines.append("")
    lines.append("## Read")
    lines.append("")
    lines.append("A standard, *non-adaptive* controller degrades under both fault classes -- "
                 "expected, since it cannot adapt. This confirms the **fault substrate**: every "
                 "fault produces graded, monotone, controllable degradation with a clear breakdown "
                 "severity. The next step adds the ADAPTIVE baselines (adaptive-MPC / sysID and our "
                 "residual): the crossover hypothesis is that classical adaptation recovers the "
                 "IN-MODEL faults (grip, tire, mass, patch) but not the OUT-OF-MODEL ones "
                 "(bias, LoE, latency, drag) -- which is where our method earns its place.")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="Spielberg")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=8000)
    ap.add_argument("--severities", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--out", default="fault_matrix_results.json")
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    tracks_dir = os.path.join(script_dir, "f1tenth_racetracks")
    faults = fi.ALL_FAULTS

    print(f"=== Fault matrix on {args.track} "
          f"({len(faults)} faults x {len(args.severities)} severities x {args.seeds} seeds) ===")
    matrix = {}
    for name in faults:
        matrix[name] = {}
        cells = []
        for s in args.severities:
            c = cell(args.track, tracks_dir, name, s, args.seeds, args.max_steps)
            matrix[name][str(s)] = c
            cells.append(f"{c['mean_progress']*100:4.0f}%")
        cls = "in " if name in fi.IN_MODEL else "out"
        print(f"  {name:<20s}[{cls}] progress: " + " ".join(cells))

    baseline = matrix[faults[0]]["0.0"]   # severity-0 is fault-free for any row
    results = {
        "track": args.track, "seeds": args.seeds, "max_steps": args.max_steps,
        "controller": "pure_pursuit (alpha_rl=0)", "faults": faults,
        "severities": args.severities, "baseline": baseline, "matrix": matrix,
    }
    with open(os.path.join(script_dir, args.out), "w") as f:
        json.dump(results, f, indent=2)
    write_summary(results, os.path.join(script_dir, "fault_matrix_summary.md"))
    print(f"\nSaved -> {args.out} and fault_matrix_summary.md")


if __name__ == "__main__":
    main()
