"""Train a residual on top of a RACELINE-referenced Pure Pursuit and measure
whether it holds completion while cutting deviation from the true raceline.

Compares pp_blend baselines: the residual now makes SMALL, same-objective
corrections (keep PP on the racing line) instead of fighting a centerline base.
Reward is unchanged -- because the PP path is now the raceline, the existing
deviation term already rewards raceline tracking.

Multi-track CrossQ residual (geometry diversity matters -- see ablation).
Eval: held-out Austin/Monza/Silverstone + Spielberg; completion, mean|d_rl|,
lap time, all at nominal grip.
"""
import argparse, gc, json, os, time, types
import numpy as np
from sbx import CrossQ
from stable_baselines3.common.vec_env import SubprocVecEnv
from residual_env import RLPPEnv
from multi_track_env import MultiTrackEnv

HELD_OUT = ["Austin", "Monza", "Silverstone"]
EVAL_TRACKS = ["Austin", "Monza", "Silverstone", "Spielberg"]
BASE_KW = dict(alpha_rl=1.0, velocity_gain=0.5, velocity_curriculum=True,
               action_scaling=(0.05, 1.0))


def discover(tracks_dir):
    out = []
    for name in sorted(os.listdir(tracks_dir)):
        td = os.path.join(tracks_dir, name)
        if not os.path.isdir(td): continue
        rp = os.path.join(td, f"{name}_raceline.csv")
        cp = os.path.join(td, f"{name}_centerline.csv")
        mp = os.path.join(td, f"{name}_map.png")
        if not (os.path.exists(rp) and os.path.exists(cp) and os.path.exists(mp)):
            continue
        try:
            np.loadtxt(rp, delimiter=";", skiprows=3)
            np.loadtxt(cp, delimiter=",", skiprows=1)
        except Exception:
            continue
        out.append(name)
    return out


def d_rl(env, px, py):
    i = int(np.argmin((env.real_rl_xy[:, 0] - px) ** 2
                      + (env.real_rl_xy[:, 1] - py) ** 2))
    rp = env.real_rl_xy[i]; rpsi = env.real_rl_psi[i]
    return abs(-np.sin(rpsi) * (px - rp[0]) + np.cos(rpsi) * (py - rp[1]))


def make_multi(pool, tracks_dir, ref, blend, seed):
    def _init():
        e = MultiTrackEnv(real_tracks=pool, tracks_dir=tracks_dir,
                          prob_generated=0.0, pregenerate=False,
                          pp_reference=ref, pp_blend=blend, mu_noise_std=0.15,
                          **BASE_KW)
        e.reset(seed=seed)
        return e
    return _init


def evaluate(model, track, tracks_dir, ref, blend, n_laps=3):
    env = RLPPEnv(track_name=track, tracks_dir=tracks_dir, pp_reference=ref,
                  pp_blend=blend, mu_noise_std=0.0, velocity_curriculum=False,
                  max_laps=n_laps, **{k: v for k, v in BASE_KW.items()
                                      if k != "velocity_curriculum"})
    completed = 0; devs = []; times = []
    for _ in range(n_laps):
        obs, _ = env.reset()
        steps = 0; cumul_s = 0.0
        prev_s = env.raceline_s[env._closest_idx]
        while True:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            steps += 1
            devs.append(d_rl(env, env._last_pos[0], env._last_pos[1]))
            cur_s = env.raceline_s[env._closest_idx]
            ds = cur_s - prev_s
            if ds < -env.total_s / 2: ds += env.total_s
            elif ds > env.total_s / 2: ds -= env.total_s
            cumul_s += ds; prev_s = cur_s
            if cumul_s >= env.total_s and steps > 100:
                completed += 1; times.append(steps * env.controller_dt); break
            if term or trunc or steps > 20000: break
    env.close()
    return {"completed": completed, "n_laps": n_laps,
            "mean_d_rl": float(np.mean(devs)) if devs else None,
            "mean_time": float(np.mean(times)) if times else None}


def run_cfg(ref, blend, seed, args, tracks_dir, pool):
    t0 = time.time()
    venv = SubprocVecEnv([make_multi(pool, tracks_dir, ref, blend, seed * 100 + i)
                          for i in range(args.envs)])
    model = CrossQ("MlpPolicy", venv, learning_rate=args.lr,
                   buffer_size=100_000, batch_size=256, gamma=0.99,
                   gradient_steps=args.utd, learning_starts=256,
                   policy_kwargs=dict(net_arch=[256, 256]), verbose=0, seed=seed)
    tag = f"ref={ref} blend={blend} seed={seed}"
    print(f"[{tag}] training {args.steps} steps...", flush=True)
    model.learn(total_timesteps=args.steps)
    venv.close()
    cells = {}
    for track in EVAL_TRACKS:
        res = evaluate(model, track, tracks_dir, ref, blend, args.eval_laps)
        cells[track] = res
        print(f"    {tag} {track:<12s} {res['completed']}/{res['n_laps']} "
              f"|d_rl|={res['mean_d_rl']:.4f} t={res['mean_time']}", flush=True)
    del model; gc.collect()
    print(f"[{tag}] done {(time.time()-t0)/60:.1f}m", flush=True)
    return {"ref": ref, "blend": blend, "seed": seed, "cells": cells}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=80_000)
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--utd", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-laps", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    # configs as ref:blend pairs
    ap.add_argument("--out", default="train_residual_ref_results.json")
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    tracks_dir = os.path.join(script_dir, "f1tenth_racetracks")
    pool = [t for t in discover(tracks_dir) if t not in HELD_OUT]

    configs = [("centerline", 0.0), ("raceline", 0.75)]
    print("=== Residual on raceline-referenced PP ===")
    print(f"  steps={args.steps} utd={args.utd} envs={args.envs} seeds={args.seeds}")
    print(f"  configs={configs}")
    print(f"  pool({len(pool)}) eval={EVAL_TRACKS}", flush=True)

    results = []; t0 = time.time()
    for ref, blend in configs:
        for seed in args.seeds:
            r = run_cfg(ref, blend, seed, args, tracks_dir, pool)
            results.append(r)
            with open(args.out, "w") as f:
                json.dump({"args": vars(args), "results": results}, f, indent=2)
            print(f"  [ckpt] {len(results)} runs, {(time.time()-t0)/60:.1f}m\n", flush=True)
    print(f"=== DONE {(time.time()-t0)/60:.1f}m -> {args.out} ===")


if __name__ == "__main__":
    main()
