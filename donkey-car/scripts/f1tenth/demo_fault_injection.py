"""Phase 0 face-validity demo for fault injection (resilient-adaptation pivot).

Runs a STANDARD controller (Pure Pursuit alone, alpha_rl=0 -- the residual is
zeroed) on one track, first clean, then under the three headline faults, then
sweeps severity. Shows that:
  1. the sim supports fault injection through existing hooks,
  2. a standard, community-accepted controller drives the track clean,
  3. each fault reproducibly degrades it, with a monotone severity knob.

Output: fault_demo_results.json  (+ fault_demo_traces.png if matplotlib is present)

Run:
  python demo_fault_injection.py --track Spielberg
"""

import argparse
import json
import os
import numpy as np

from residual_env import RLPPEnv
import fault_injection as fi


# Standard-controller config: Pure Pursuit alone (residual zeroed), deterministic.
BASE_KW = dict(
    alpha_rl=0.0,               # pure Pure Pursuit, no learned residual
    velocity_gain=0.5,          # the operating point where PP completes laps alone
    mu_noise_std=0.0,           # deterministic
    velocity_curriculum=False,
    pp_reference="centerline",  # classic PP (community-standard baseline)
)

ZERO_ACTION = np.zeros(2, dtype=np.float32)


def run_episode(env, max_steps, seed=0):
    """Drive Pure Pursuit until crash / lap target / step cap. Returns metrics."""
    obs, _ = env.reset(seed=seed)
    total_s = env.total_s
    prev_s = env.raceline_s[env._closest_idx]
    cumul_s = 0.0
    devs = []
    crashed = False
    crash_step = None
    steps = 0

    for steps in range(1, max_steps + 1):
        obs, r, term, trunc, info = env.step(ZERO_ACTION)

        d, _ = env.pp.get_frenet_state(
            env._last_pos[0], env._last_pos[1], env._last_heading, env._closest_idx
        )
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
            crash_step = steps
            break
        if term or trunc:
            break

    return {
        "steps": steps,
        "crashed": crashed,
        "crash_step": crash_step,
        "laps_completed": int(max(0.0, cumul_s) // total_s),
        "progress_frac": float(max(0.0, cumul_s) / total_s),
        "mean_abs_dev": float(np.mean(devs)) if devs else None,
        "max_abs_dev": float(np.max(devs)) if devs else None,
        "final_vx": float(info["vx"]),
    }


def make_env(track, tracks_dir, faults):
    return RLPPEnv(track_name=track, tracks_dir=tracks_dir, max_laps=1,
                   faults=faults, **BASE_KW)


def summarize(tag, m):
    status = f"CRASH@{m['crash_step']}" if m["crashed"] else f"survived {m['steps']}"
    dev = f"{m['mean_abs_dev']:.3f}" if m["mean_abs_dev"] is not None else "  -  "
    print(f"  {tag:<34s} {status:<16s} progress={m['progress_frac']*100:5.1f}%  "
          f"mean|d|={dev}m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="Spielberg")
    ap.add_argument("--max-steps", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="fault_demo_results.json")
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    tracks_dir = os.path.join(script_dir, "f1tenth_racetracks")

    results = {"track": args.track, "config": {k: v for k, v in BASE_KW.items()}}

    # --- 1. Clean baseline (standard controller, no fault) --------------------
    print(f"\n=== Fault-injection face-validity demo on {args.track} ===")
    print("Standard controller = Pure Pursuit alone (alpha_rl=0), velocity_gain=0.5\n")
    print("[1] Clean baseline (no fault):")
    env = make_env(args.track, tracks_dir, None)
    base = run_episode(env, args.max_steps, args.seed)
    env.close()
    summarize("clean baseline", base)
    results["baseline"] = base

    # --- 2. Three headline faults at a representative severity -----------------
    headline = [
        ("friction_drop (in-model, anchor)", fi.from_spec("friction_drop", 0.6)),
        ("steering_bias (out-of-model)", fi.from_spec("steering_bias", 0.5)),
        ("actuator_latency (out-of-model)", fi.from_spec("actuator_latency", 0.6)),
    ]
    print("\n[2] Headline faults (severity fixed):")
    results["headline"] = {}
    for tag, fault in headline:
        env = make_env(args.track, tracks_dir, fault)
        m = run_episode(env, args.max_steps, args.seed)
        env.close()
        summarize(tag, m)
        results["headline"][fault.name] = {"mechanism": fault.mechanism, **m}

    # --- 3. Severity sweep (monotone knob check) ------------------------------
    sweep_faults = ["friction_drop", "steering_bias", "actuator_latency",
                    "steering_loe", "tire_stiffness"]
    severities = [0.0, 0.25, 0.5, 0.75, 1.0]
    print("\n[3] Severity sweep (progress % of one lap; lower = more degraded):")
    print(f"  {'fault':<20s}" + "".join(f"  s={s:<4}" for s in severities))
    results["sweep"] = {}
    for name in sweep_faults:
        row = []
        for s in severities:
            fault = None if s == 0.0 else fi.from_spec(name, s)
            env = make_env(args.track, tracks_dir, fault)
            m = run_episode(env, args.max_steps, args.seed)
            env.close()
            row.append(m["progress_frac"])
        results["sweep"][name] = {"severities": severities, "progress_frac": row}
        cls = "in " if name in fi.IN_MODEL else "out"
        print(f"  {name:<20s}[{cls}]" + "".join(f"  {p*100:5.1f}" for p in row))

    with open(os.path.join(script_dir, args.out), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.out}")

    # --- 4. Optional trajectory plot ------------------------------------------
    try:
        _plot_traces(args, tracks_dir, headline, script_dir)
    except Exception as e:
        print(f"(plot skipped: {e})")


def _plot_traces(args, tracks_dir, headline, script_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def trace(faults):
        env = make_env(args.track, tracks_dir, faults)
        obs, _ = env.reset(seed=args.seed)
        xs, ys = [], []
        for _ in range(args.max_steps):
            obs, r, term, trunc, info = env.step(ZERO_ACTION)
            xs.append(env._last_pos[0]); ys.append(env._last_pos[1])
            if info["collision"] or term or trunc:
                break
        env.close()
        return np.array(xs), np.array(ys), info["collision"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    cl = np.loadtxt(os.path.join(tracks_dir, args.track, f"{args.track}_centerline.csv"),
                    delimiter=",", skiprows=1)
    panels = [("Clean baseline", None)] + headline
    for ax, (tag, fault) in zip(axes, panels):
        ax.plot(cl[:, 0], cl[:, 1], "k--", lw=0.6, alpha=0.4)
        xs, ys, crashed = trace(fault)
        ax.plot(xs, ys, lw=1.6, color="crimson" if crashed else "seagreen")
        if crashed:
            ax.plot(xs[-1], ys[-1], "rx", ms=12, mew=3)
        ax.set_title(tag.split(" (")[0], fontsize=10)
        ax.set_aspect("equal"); ax.axis("off")
    fig.suptitle(f"Fault injection on {args.track}: standard Pure Pursuit "
                 f"(green=survived, red=crashed)", fontsize=12)
    out = os.path.join(script_dir, "fault_demo_traces.png")
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"Saved -> fault_demo_traces.png")


if __name__ == "__main__":
    main()
