"""Phase C -- Reptile meta-adaptation over FAULTS (resilient-adaptation pivot).

Forks the track-Reptile idea (train_reptile.py): instead of meta-learning an
initialization that adapts to any new TRACK, learn one that adapts to any new
FAULT (a degradation of the car / sensing) in a few laps of fine-tuning.

A task = (track, fault_type, severity), sampled from fault_distribution.py.
Meta-loop:
  1. warm-start a competent clean residual (no fault),
  2. for each meta-iteration: sample a fault task, clone meta-params into a fresh
     CrossQ, fine-tune on that fault (inner loop), Reptile-interpolate back.
At deployment: load meta-params, fine-tune a few laps on whatever fault appears.

Evaluation reports, at 0 / 2K / 4K adaptation steps:
  - held-out SEVERITIES of trained fault types  (in-distribution)
  - held-out fault TYPES never trained on        (out-of-distribution -- headline)

Reuses the Reptile mechanics (snapshot/load/interpolate, create_model,
checkpointing) directly from train_reptile.py.

Example (full run on pistar GPU):
  export PYTHONNOUSERSITE=1
  python train_reptile_faults.py --meta-iterations 200 --inner-steps 8000 \
      --inner-envs 4 --utd 20 --tag reptile_faults_v1
Smoke test (CPU, seconds):
  python train_reptile_faults.py --smoke
"""

import argparse
import gc
import os
import pickle
import shutil
import sys
import time

import numpy as np
import jax

from train_reptile import (
    snapshot_params, load_params, reptile_interpolate, create_model,
    save_meta_checkpoint, load_meta_checkpoint, discover_valid_tracks,
)
import fault_injection as fi
import fault_distribution as fd
from fault_distribution import make_fault_env_fn, sample_task


HELD_OUT_TRACKS = ["Austin", "Monza", "Silverstone"]   # keep eval geometry unseen
WARMUP_TRACKS = ["Spielberg", "Sakhir", "Spa"]


def base_env_kwargs(args):
    """Shared RLPPEnv kwargs. Faults are injected per task by the env factory."""
    return dict(
        alpha_rl=1.0,
        velocity_gain=0.5,
        mu_noise_std=0.0,          # the fault provides the perturbation, not random mu
        velocity_curriculum=True,
        action_scaling=tuple(args.action_scaling),
        pp_reference=args.pp_reference,
        pp_blend=args.pp_blend,
    )


def eval_fault_run(model, track, fault_name, severity, tracks_dir, env_kwargs,
                   n_laps=3, step_cap=20000):
    """Deterministic rollout under a fixed fault: laps completed + mean progress."""
    from residual_env import RLPPEnv
    fault = fi.from_spec(fault_name, severity) if fault_name else None
    ek = dict(env_kwargs)
    ek["velocity_curriculum"] = False
    env = RLPPEnv(track_name=track, tracks_dir=tracks_dir, faults=fault,
                  max_laps=n_laps, **ek)
    completed = 0
    progs = []
    for _ in range(n_laps):
        obs, _ = env.reset()
        total_s = env.total_s
        prev_s = env.raceline_s[env._closest_idx]
        cumul = 0.0
        steps = 0
        while True:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            steps += 1
            cur = env.raceline_s[env._closest_idx]
            ds = cur - prev_s
            if ds < -total_s / 2:
                ds += total_s
            elif ds > total_s / 2:
                ds -= total_s
            cumul += ds
            prev_s = cur
            if cumul >= total_s and steps > 100:
                completed += 1
                break
            if term or trunc or steps > step_cap:
                break
        progs.append(min(max(cumul, 0.0) / total_s, 1.0))
    env.close()
    return completed, float(np.mean(progs)) if progs else 0.0


def evaluate_fault_adaptation(meta_params, tasks, tracks_dir, args, adapt_steps_list,
                              env_kwargs, n_laps=3, step_cap=20000):
    """Fine-tune meta-init on each held-out task for N steps; report recovery."""
    from stable_baselines3.common.vec_env import DummyVecEnv
    out = {}
    for (track, name, sev) in tasks:
        key = f"{name}@{sev:.2f}"
        out[key] = {}
        for adapt in adapt_steps_list:
            env = DummyVecEnv([make_fault_env_fn(track, tracks_dir, name, sev,
                                                 seed=args.seed + 300, **env_kwargs)])
            model = create_model(env, args)
            load_params(model, meta_params)
            if adapt > 0:
                model.learn(total_timesteps=adapt)
            c, p = eval_fault_run(model, track, name, sev, tracks_dir, env_kwargs,
                                  n_laps=n_laps, step_cap=step_cap)
            out[key][adapt] = {"completed": c, "progress": p}
            env.close()
            del model
            gc.collect()
    return out


def build_eval_tasks(eval_track):
    """Held-out tasks: in-distribution (trained types, held-out severity) and
    out-of-distribution (never-trained fault types)."""
    id_tasks = [(eval_track, n, 0.5) for n in fd.TRAIN_FAULTS[:4]]
    ood_tasks = [(eval_track, n, 0.4) for n in fd.HELD_OUT_FAULTS]
    return id_tasks, ood_tasks


def warm_start(args, tracks_dir, env_kwargs):
    """Train a competent clean (fault-free) residual to initialize the meta-params."""
    from stable_baselines3.common.vec_env import DummyVecEnv
    wt = next((t for t in WARMUP_TRACKS if t in args._train_tracks), args._train_tracks[0])
    print(f"Warm-starting (clean, no fault) on {wt} for {args.warmup_steps} steps...")
    env = DummyVecEnv([make_fault_env_fn(wt, tracks_dir, None, 0.0, seed=args.seed,
                                         **env_kwargs)])
    model = create_model(env, args)
    model.learn(total_timesteps=args.warmup_steps)
    meta_params = snapshot_params(model)
    env.close()
    del model
    gc.collect()
    return meta_params


def main():
    ap = argparse.ArgumentParser(description="Reptile meta-adaptation over faults (Phase C)")
    ap.add_argument("--meta-iterations", type=int, default=200)
    ap.add_argument("--inner-steps", type=int, default=8000)
    ap.add_argument("--inner-envs", type=int, default=4)
    ap.add_argument("--utd", type=int, default=20)
    ap.add_argument("--inner-lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--epsilon-start", type=float, default=1.0)
    ap.add_argument("--epsilon-end", type=float, default=0.1)
    ap.add_argument("--warmup-steps", type=int, default=50_000)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--eval-track", default="Austin")
    ap.add_argument("--action-scaling", type=float, nargs=2, default=[0.05, 1.0])
    ap.add_argument("--pp-reference", default="centerline", choices=["centerline", "raceline"])
    ap.add_argument("--pp-blend", type=float, default=1.0)
    ap.add_argument("--tag", default="reptile_faults_v1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny end-to-end run to verify the pipeline (seconds).")
    args = ap.parse_args()

    if args.smoke:
        args.meta_iterations = 2
        args.inner_steps = 400
        args.inner_envs = 2
        args.warmup_steps = 400
        args.eval_every = 2
        args.save_every = 2

    script_dir = os.path.dirname(os.path.abspath(__file__))
    tracks_dir = os.path.join(script_dir, "f1tenth_racetracks")
    save_dir = os.path.join(script_dir, "reptile", args.tag)

    all_tracks = discover_valid_tracks(tracks_dir)
    train_tracks = [t for t in all_tracks if t not in HELD_OUT_TRACKS]
    if not train_tracks:
        raise SystemExit("no training tracks discovered")
    args._train_tracks = train_tracks
    env_kwargs = base_env_kwargs(args)

    print(f"=== Reptile-over-faults: {args.tag} ===")
    print(f"  meta-iters={args.meta_iterations} inner-steps={args.inner_steps} "
          f"envs={args.inner_envs} utd={args.utd}")
    print(f"  train fault types ({len(fd.TRAIN_FAULTS)}): {fd.TRAIN_FAULTS}")
    print(f"  held-out (OOD) fault types: {fd.HELD_OUT_FAULTS}")
    print(f"  severity band: {fd.SEV_RANGE}   train tracks: {len(train_tracks)}")
    print(f"  eval track: {args.eval_track}   PP ref: {args.pp_reference} blend={args.pp_blend}")

    if args.resume:
        from stable_baselines3.common.vec_env import DummyVecEnv
        env = DummyVecEnv([make_fault_env_fn(train_tracks[0], tracks_dir, None, 0.0,
                                             seed=args.seed, **env_kwargs)])
        meta_model = create_model(env, args)
        meta_model.learn(total_timesteps=args.batch_size * 2)
        meta_params = load_meta_checkpoint(args.resume)
        load_params(meta_model, meta_params)
        env.close()
        del meta_model
        gc.collect()
        print(f"resumed from {args.resume}")
    else:
        meta_params = warm_start(args, tracks_dir, env_kwargs)

    id_tasks, ood_tasks = build_eval_tasks(args.eval_track)
    if args.smoke:
        id_tasks, ood_tasks = id_tasks[:1], ood_tasks[:1]
    eval_adapt = [0, 200] if args.smoke else [0, 2000, 4000]
    eval_laps = 1 if args.smoke else 3
    eval_cap = 2000 if args.smoke else 20000
    rng = np.random.RandomState(args.seed)
    os.makedirs(save_dir, exist_ok=True)
    t0 = time.time()
    best_score = -1.0

    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    VecEnv = DummyVecEnv if (args.smoke or args.inner_envs == 1) else SubprocVecEnv

    for it in range(args.meta_iterations):
        it_t0 = time.time()
        frac = it / max(args.meta_iterations - 1, 1)
        epsilon = args.epsilon_start + frac * (args.epsilon_end - args.epsilon_start)

        track, fault_name, sev = sample_task(rng, train_tracks)
        env = VecEnv([make_fault_env_fn(track, tracks_dir, fault_name, sev,
                                        seed=args.seed + it * 100 + i, **env_kwargs)
                      for i in range(args.inner_envs)])
        inner = create_model(env, args)
        load_params(inner, meta_params)
        inner.learn(total_timesteps=args.inner_steps)
        adapted = snapshot_params(inner)
        meta_params = reptile_interpolate(meta_params, adapted, epsilon)
        env.close()
        del inner, adapted
        gc.collect()
        if (it + 1) % 10 == 0:
            jax.clear_caches()

        print(f"  [{it+1}/{args.meta_iterations}] {fault_name}@{sev:.2f} on {track:<14s} "
              f"eps={epsilon:.3f} iter={time.time()-it_t0:.0f}s "
              f"eta={((time.time()-t0)/(it+1))*(args.meta_iterations-it-1)/60:.0f}m")
        sys.stdout.flush()

        # checkpoint latest
        params_np = jax.tree.map(lambda x: np.array(x), meta_params)
        with open(os.path.join(save_dir, "meta_params_latest.pkl"), "wb") as f:
            pickle.dump(params_np, f)
        if (it + 1) % args.save_every == 0:
            save_meta_checkpoint(meta_params, it + 1, save_dir)

        if (it + 1) % args.eval_every == 0:
            print(f"  === eval @ iter {it+1} (adapt {'/'.join(map(str, eval_adapt))}) ===")
            id_res = evaluate_fault_adaptation(meta_params, id_tasks, tracks_dir, args,
                                               eval_adapt, env_kwargs,
                                               n_laps=eval_laps, step_cap=eval_cap)
            ood_res = evaluate_fault_adaptation(meta_params, ood_tasks, tracks_dir, args,
                                                eval_adapt, env_kwargs,
                                                n_laps=eval_laps, step_cap=eval_cap)
            score = 0
            last = eval_adapt[-1]
            for label, res in (("ID", id_res), ("OOD", ood_res)):
                for key, cells in res.items():
                    summ = " ".join(f"{a}:{cells[a]['completed']}" for a in eval_adapt)
                    print(f"    [{label}] {key:<22s} {summ} "
                          f"(prog@{last} {cells[last]['progress']*100:.0f}%)")
                    score += cells[last]["completed"]
            if score > best_score:
                best_score = score
                bp = save_meta_checkpoint(meta_params, it + 1, save_dir)
                shutil.copy2(bp, os.path.join(save_dir, "meta_params_best.pkl"))
                print(f"    new best: {score} completed laps -> meta_params_best.pkl")
            sys.stdout.flush()

    print(f"\n=== done in {(time.time()-t0)/60:.1f}m; best score {best_score} ===")


if __name__ == "__main__":
    main()
