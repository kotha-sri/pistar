"""Reptile meta-learning for few-lap track adaptation (v3).

Trains a meta-initialization of CrossQ that can adapt to any new track
in 1-5 laps (~4K-10K env steps) of fine-tuning.

Reptile (Nichol et al., 2018):
  for each meta-iteration:
    1. Sample a track (task)
    2. Clone meta-params into a fresh CrossQ
    3. Fine-tune on that track for K steps (inner loop)
    4. meta_params += epsilon * (adapted_params - meta_params)

At deployment: load meta-params, fine-tune on new track for a few laps.

v3 changes from v2:
  - Don't meta-learn BatchNorm stats — only interpolate network weights.
    Batch stats are track-distribution-specific; averaging them across 68
    tracks produces a useless prior. Stats from the warm-start remain as
    the fixed prior and are updated during per-track adaptation normally.
  - Restore SubprocVecEnv with 4 workers for wall-clock speed (~80s/iter).
    v2's OOM was caused by 50K buffer per worker and unbounded JAX cache,
    not the worker count itself. Fixed by: buffer capped at inner_steps
    (20K), gc.collect() + jax.clear_caches() every 50 iters.
  - Warm-start meta-params by training on easy tracks (Spielberg + Sakhir)
    for warmup_steps before the meta-loop, instead of near-random init.
  - gc.collect() + jax.clear_caches() every 50 iters to prevent OOM.
  - velocity_gain=0.5 throughout (training + eval). At 0.75 PP crashes in
    97 steps, making the reward signal too sparse for CrossQ to learn at
    any budget. At 0.5 PP completes laps alone; RL only needs to learn the
    residual correction, making 4K-10K adaptation meaningful.
  - Default iterations increased to 400.
"""

import argparse
import gc
import os
import shutil
import sys
import time
import pickle

import numpy as np
import jax
import jax.numpy as jnp
from sbx import CrossQ
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from residual_env import RLPPEnv


HELD_OUT_TRACKS = ["Austin", "Monza", "Silverstone"]
# Tracks to warm-start on — must exist in f1tenth_racetracks/
WARMUP_TRACKS = ["Spielberg", "Sakhir", "Spa"]


def discover_valid_tracks(tracks_dir):
    """Find tracks with complete, well-formed data files."""
    tracks = []
    if not os.path.isdir(tracks_dir):
        return tracks
    for name in sorted(os.listdir(tracks_dir)):
        td = os.path.join(tracks_dir, name)
        if not os.path.isdir(td):
            continue
        raceline_path = os.path.join(td, f"{name}_raceline.csv")
        centerline_path = os.path.join(td, f"{name}_centerline.csv")
        map_path = os.path.join(td, f"{name}_map.png")
        if not (os.path.exists(raceline_path) and os.path.exists(centerline_path)
                and os.path.exists(map_path)):
            continue
        try:
            np.loadtxt(raceline_path, delimiter=";", skiprows=3)
            np.loadtxt(centerline_path, delimiter=",", skiprows=1)
        except (ValueError, OSError):
            continue
        tracks.append(name)
    return tracks


def snapshot_params(model):
    """Capture all network state for save/load and Reptile interpolation."""
    p = model.policy
    return {
        "actor_params": jax.tree.map(jnp.copy, p.actor_state.params),
        "actor_batch_stats": jax.tree.map(jnp.copy, p.actor_state.batch_stats),
        "qf_params": jax.tree.map(jnp.copy, p.qf_state.params),
        "qf_batch_stats": jax.tree.map(jnp.copy, p.qf_state.batch_stats),
        "ent_coef_params": jax.tree.map(jnp.copy, model.ent_coef_state.params),
    }


def load_params(model, snap):
    """Restore all network state from a snapshot."""
    p = model.policy
    p.actor_state = p.actor_state.replace(
        params=snap["actor_params"],
        batch_stats=snap["actor_batch_stats"],
    )
    p.qf_state = p.qf_state.replace(
        params=snap["qf_params"],
        batch_stats=snap["qf_batch_stats"],
    )
    model.ent_coef_state = model.ent_coef_state.replace(
        params=snap["ent_coef_params"],
    )


# Keys to interpolate in the Reptile update.
# BatchNorm stats are excluded: they encode the observation distribution of
# the specific track just trained, so interpolating them across all tracks
# produces a useless average. The warm-start batch_stats serve as a fixed
# prior; they update naturally during per-track adaptation.
_WEIGHT_KEYS = {"actor_params", "qf_params", "ent_coef_params"}


def reptile_interpolate(meta, adapted, epsilon):
    """Reptile update: interpolate weight keys only, freeze batch_stats."""
    return {
        k: (jax.tree.map(lambda m, a: m + epsilon * (a - m), meta[k], adapted[k])
            if k in _WEIGHT_KEYS else meta[k])
        for k in meta
    }


def make_env_fn(track_name, tracks_dir, seed=None, **env_kwargs):
    def _init():
        env = RLPPEnv(track_name=track_name, tracks_dir=tracks_dir, **env_kwargs)
        if seed is not None:
            env.reset(seed=seed)
        return Monitor(env)
    return _init


def create_model(env, args):
    # Buffer capped at inner_steps — enough to hold one full inner loop's data.
    # v2 used 50K which was unnecessarily large and contributed to OOM.
    return CrossQ(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.inner_lr,
        buffer_size=max(args.inner_steps, 20_000),
        batch_size=args.batch_size,
        gamma=0.99,
        gradient_steps=args.utd,
        learning_starts=args.batch_size,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=0,
        seed=args.seed,
    )


def evaluate_on_track(model, track_name, tracks_dir, n_laps=10, alpha_rl=0.55,
                      action_scaling=(0.05, 1.0), pp_reference="centerline",
                      pp_blend=1.0):
    env = RLPPEnv(
        track_name=track_name, tracks_dir=tracks_dir,
        alpha_rl=alpha_rl, velocity_gain=0.5, action_scaling=action_scaling,
        mu_noise_std=0.0, velocity_curriculum=False, max_laps=n_laps,
        pp_reference=pp_reference, pp_blend=pp_blend,
    )
    lap_times = []
    for _ in range(n_laps):
        obs, _ = env.reset()
        env.alpha_rl = alpha_rl
        steps = 0
        cumul_s = 0.0
        prev_s = env.raceline_s[env._closest_idx]
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            cur_s = env.raceline_s[env._closest_idx]
            ds = cur_s - prev_s
            if ds < -env.total_s / 2:
                ds += env.total_s
            elif ds > env.total_s / 2:
                ds -= env.total_s
            cumul_s += ds
            prev_s = cur_s
            if cumul_s >= env.total_s and steps > 100:
                lap_times.append(steps * env.controller_dt)
                break
            if terminated or truncated or steps > 20000:
                break
    return lap_times


def evaluate_meta_adaptation(meta_params, track_name, tracks_dir, args,
                              adapt_steps_list=None, action_scaling=(0.05, 1.0)):
    """Evaluate meta-init by fine-tuning for various step counts."""
    if adapt_steps_list is None:
        adapt_steps_list = [0, 4000, 10000]

    env_kwargs = dict(
        alpha_rl=1.0,
        velocity_gain=0.5,
        mu_noise_std=0.15,
        velocity_curriculum=True,
        action_scaling=action_scaling,
        pp_reference=args.pp_reference,
        pp_blend=args.pp_blend,
    )

    results = {}
    for adapt_steps in adapt_steps_list:
        env = DummyVecEnv([make_env_fn(
            track_name, tracks_dir, seed=args.seed + 200, **env_kwargs,
        )])
        model = create_model(env, args)
        load_params(model, meta_params)

        if adapt_steps > 0:
            model.learn(total_timesteps=adapt_steps)

        laps = evaluate_on_track(model, track_name, tracks_dir,
                                action_scaling=action_scaling,
                                pp_reference=args.pp_reference,
                                pp_blend=args.pp_blend)
        results[adapt_steps] = {
            "completed": len(laps),
            "mean_time": np.mean(laps) if laps else None,
        }
        env.close()
        del model
        gc.collect()

    return results


def save_meta_checkpoint(meta_params, meta_iter, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    params_np = jax.tree.map(lambda x: np.array(x), meta_params)
    path = os.path.join(save_dir, f"meta_params_{meta_iter:04d}.pkl")
    with open(path, "wb") as f:
        pickle.dump(params_np, f)
    latest = os.path.join(save_dir, "meta_params_latest.pkl")
    with open(latest, "wb") as f:
        pickle.dump(params_np, f)
    return path


def load_meta_checkpoint(path):
    with open(path, "rb") as f:
        params_np = pickle.load(f)
    return jax.tree.map(jnp.array, params_np)


def main():
    parser = argparse.ArgumentParser(description="Reptile meta-learning for few-lap adaptation (v3)")
    parser.add_argument("--meta-iterations", type=int, default=400)
    parser.add_argument("--inner-steps", type=int, default=20_000)
    parser.add_argument("--inner-envs", type=int, default=4)
    parser.add_argument("--utd", type=int, default=20)
    parser.add_argument("--inner-lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=50_000,
                        help="Steps to pre-train on easy tracks before Reptile loop")
    parser.add_argument("--include-generated", action="store_true",
                        help="Include procedurally generated tracks in training pool")
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--tag", default="reptile_v3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None, help="Path to meta checkpoint to resume from")
    parser.add_argument("--reward-alpha-raceline", type=float, default=0.0,
                        help="Raceline proximity reward weight (0=off)")
    parser.add_argument("--reward-use-raceline-progress", action="store_true",
                        help="Track progress along real raceline instead of centerline")
    parser.add_argument("--reward-alpha-dev", type=float, default=1.0,
                        help="Centerline deviation penalty weight (default 1.0)")
    parser.add_argument("--action-scaling", type=float, nargs=2, default=[0.05, 1.0],
                        metavar=("STEER", "VEL"),
                        help="Action scaling for [steering, velocity] residuals")
    parser.add_argument("--pp-reference", default="centerline",
                        choices=["centerline", "raceline"],
                        help="Path PP tracks: centerline (old) or feasible raceline")
    parser.add_argument("--pp-blend", type=float, default=1.0,
                        help="centerline(0)->raceline(1) blend when pp-reference=raceline")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    tracks_dir = os.path.join(script_dir, "f1tenth_racetracks")
    gen_tracks_dir = os.path.join(script_dir, "generated_tracks")
    save_dir = os.path.join(script_dir, "reptile", args.tag)

    all_real_tracks = discover_valid_tracks(tracks_dir)
    train_tracks = [(t, tracks_dir) for t in all_real_tracks if t not in HELD_OUT_TRACKS]

    if args.include_generated:
        gen_tracks = discover_valid_tracks(gen_tracks_dir)
        train_tracks.extend([(t, gen_tracks_dir) for t in gen_tracks])

    train_names = [t[0] for t in train_tracks]

    print(f"=== Reptile Meta-Learning v3: {args.tag} ===")
    print(f"  Meta iterations: {args.meta_iterations}")
    print(f"  Inner steps: {args.inner_steps} ({args.inner_envs} envs, UTD={args.utd})")
    print(f"  BatchNorm: warm-start prior only, NOT meta-learned")
    print(f"  Warm-start: {args.warmup_steps} steps on {WARMUP_TRACKS}")
    print(f"  Epsilon: {args.epsilon_start} -> {args.epsilon_end}")
    print(f"  Training tracks ({len(train_tracks)}): {train_names[:10]}{'...' if len(train_names) > 10 else ''}")
    print(f"  Held-out eval tracks: {HELD_OUT_TRACKS}")
    if args.include_generated:
        print(f"  Generated tracks included: {len(gen_tracks)}")
    if args.reward_alpha_raceline > 0:
        print(f"  Raceline reward: alpha={args.reward_alpha_raceline}")
    if args.reward_use_raceline_progress:
        print(f"  Using raceline progress tracking")

    reward_overrides = {}
    if args.reward_alpha_raceline > 0:
        reward_overrides["alpha_raceline"] = args.reward_alpha_raceline
    if args.reward_use_raceline_progress:
        reward_overrides["use_raceline_progress"] = True
    if args.reward_alpha_dev != 1.0:
        reward_overrides["alpha_dev"] = args.reward_alpha_dev

    action_scaling = tuple(args.action_scaling)
    print(f"  Action scaling: steer={action_scaling[0]}, vel={action_scaling[1]}")

    env_kwargs = dict(
        alpha_rl=1.0,
        velocity_gain=0.5,
        mu_noise_std=0.15,
        velocity_curriculum=True,
        action_scaling=action_scaling,
        reward_overrides=reward_overrides if reward_overrides else None,
        pp_reference=args.pp_reference,
        pp_blend=args.pp_blend,
    )
    print(f"  PP reference: {args.pp_reference} (blend={args.pp_blend})")

    if args.resume:
        print(f"\nResuming from {args.resume}")
        # Need a dummy model to set up the architecture, then load checkpoint.
        init_track, init_dir = train_tracks[0]
        init_env = DummyVecEnv([make_env_fn(init_track, init_dir, seed=args.seed, **env_kwargs)])
        meta_model = create_model(init_env, args)
        meta_model.learn(total_timesteps=args.batch_size * 2)
        meta_params = load_meta_checkpoint(args.resume)
        load_params(meta_model, meta_params)
        init_env.close()
        del meta_model
        gc.collect()
    else:
        # Warm-start: train on easy tracks to get a good initial policy.
        # Each warmup track trains for warmup_steps, sequentially.
        warmup_tracks_avail = [t for t in WARMUP_TRACKS if t in train_names]
        if not warmup_tracks_avail:
            warmup_tracks_avail = [train_tracks[0][0]]
        print(f"\nWarm-starting on {warmup_tracks_avail} ({args.warmup_steps} steps each)...")
        init_env = DummyVecEnv([make_env_fn(
            warmup_tracks_avail[0], tracks_dir, seed=args.seed, **env_kwargs,
        )])
        meta_model = create_model(init_env, args)
        meta_model.learn(total_timesteps=args.warmup_steps)
        meta_params = snapshot_params(meta_model)
        init_env.close()
        del meta_model
        gc.collect()

        for wt in warmup_tracks_avail[1:]:
            print(f"  -> Continuing warm-start on {wt}...")
            wenv = DummyVecEnv([make_env_fn(wt, tracks_dir, seed=args.seed + 1, **env_kwargs)])
            wmodel = create_model(wenv, args)
            load_params(wmodel, meta_params)
            wmodel.learn(total_timesteps=args.warmup_steps)
            # Average weights from this track into meta_params (simple ensemble warm-start).
            # We DON'T use reptile_interpolate here — just a direct snapshot to keep the
            # batch_stats from the last warm-start track as our fixed prior.
            adapted = snapshot_params(wmodel)
            meta_params = {
                k: (jax.tree.map(lambda m, a: 0.5 * m + 0.5 * a, meta_params[k], adapted[k])
                    if k in _WEIGHT_KEYS else adapted[k])
                for k in meta_params
            }
            wenv.close()
            del wmodel
            gc.collect()

        print("  Warm-start complete.\n")

    rng = np.random.RandomState(args.seed)
    t0 = time.time()
    best_eval_score = -1

    print(f"{'='*60}")
    print(f"Starting Reptile training...")
    print(f"{'='*60}\n")

    for meta_iter in range(args.meta_iterations):
        iter_t0 = time.time()

        frac = meta_iter / max(args.meta_iterations - 1, 1)
        epsilon = args.epsilon_start + frac * (args.epsilon_end - args.epsilon_start)

        track_idx = rng.randint(len(train_tracks))
        track_name, track_dir = train_tracks[track_idx]

        # SubprocVecEnv for wall-clock speed: 4 workers parallelize env stepping,
        # cutting iter time from ~265s (1 env) back to ~80s. OOM is prevented by
        # the smaller buffer (20K not 50K) and jax.clear_caches() every 50 iters.
        inner_env = SubprocVecEnv([
            make_env_fn(track_name, track_dir,
                        seed=args.seed + meta_iter * 100 + i, **env_kwargs)
            for i in range(args.inner_envs)
        ])
        inner_model = create_model(inner_env, args)
        load_params(inner_model, meta_params)

        inner_model.learn(total_timesteps=args.inner_steps)

        adapted_params = snapshot_params(inner_model)
        meta_params = reptile_interpolate(meta_params, adapted_params, epsilon)

        inner_reward = None
        try:
            from stable_baselines3.common.utils import safe_mean
            if len(inner_model.ep_info_buffer) > 0:
                inner_reward = safe_mean([ep['r'] for ep in inner_model.ep_info_buffer])
        except Exception:
            pass

        inner_env.close()
        del inner_model
        gc.collect()

        del adapted_params
        gc.collect()
        if (meta_iter + 1) % 10 == 0:
            jax.clear_caches()

        iter_time = time.time() - iter_t0
        elapsed = time.time() - t0
        eta = (elapsed / (meta_iter + 1)) * (args.meta_iterations - meta_iter - 1)

        rew_str = f"{inner_reward:.0f}" if inner_reward is not None else "?"
        print(f"  [{meta_iter+1}/{args.meta_iterations}] "
              f"track={track_name:<20s} eps={epsilon:.3f} "
              f"inner_rew={rew_str} "
              f"iter={iter_time:.0f}s eta={eta/60:.0f}m")
        sys.stdout.flush()

        params_np = jax.tree.map(lambda x: np.array(x), meta_params)
        os.makedirs(save_dir, exist_ok=True)
        latest = os.path.join(save_dir, "meta_params_latest.pkl")
        with open(latest, "wb") as f:
            pickle.dump(params_np, f)
        if (meta_iter + 1) % args.save_every == 0:
            path = save_meta_checkpoint(meta_params, meta_iter + 1, save_dir)
            print(f"    Saved checkpoint: {path}")
            sys.stdout.flush()

        if (meta_iter + 1) % args.eval_every == 0:
            print(f"\n  === Evaluation at meta-iter {meta_iter+1} ===")
            total_completed = 0
            for eval_track in HELD_OUT_TRACKS:
                results = evaluate_meta_adaptation(
                    meta_params, eval_track, tracks_dir, args,
                    adapt_steps_list=[0, 4000, 10000],
                    action_scaling=action_scaling,
                )
                print(f"    {eval_track}:")
                for steps, res in sorted(results.items()):
                    c = res["completed"]
                    t_str = f"{res['mean_time']:.2f}s" if res["mean_time"] else "N/A"
                    print(f"      {steps:>6d} adapt steps: {c}/10 laps, mean={t_str}")
                    if steps == 10000:
                        total_completed += c

            for check_track in ["Spielberg", "Sakhir"]:
                if check_track in [t[0] for t in train_tracks]:
                    results = evaluate_meta_adaptation(
                        meta_params, check_track, tracks_dir, args,
                        adapt_steps_list=[0, 4000],
                        action_scaling=action_scaling,
                    )
                    print(f"    {check_track} (train):")
                    for steps, res in sorted(results.items()):
                        c = res["completed"]
                        t_str = f"{res['mean_time']:.2f}s" if res["mean_time"] else "N/A"
                        print(f"      {steps:>6d} adapt steps: {c}/10 laps, mean={t_str}")

            if total_completed > best_eval_score:
                best_eval_score = total_completed
                best_path = save_meta_checkpoint(meta_params, meta_iter + 1, save_dir)
                best_file = os.path.join(save_dir, "meta_params_best.pkl")
                shutil.copy2(best_path, best_file)
                print(f"    New best! {total_completed}/30 total laps -> {best_file}")
            print()

    total_time = time.time() - t0
    print(f"\n=== Reptile training complete ===")
    print(f"  Wall clock: {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"  Meta iterations: {args.meta_iterations}")
    print(f"  Best eval score: {best_eval_score}/30 laps")

    print(f"\n=== Final evaluation (all adaptation budgets) ===")
    for eval_track in HELD_OUT_TRACKS:
        results = evaluate_meta_adaptation(
            meta_params, eval_track, tracks_dir, args,
            adapt_steps_list=[0, 2000, 4000, 10000, 25000],
            action_scaling=action_scaling,
        )
        print(f"\n  {eval_track}:")
        for steps, res in sorted(results.items()):
            c = res["completed"]
            t_str = f"{res['mean_time']:.2f}s" if res["mean_time"] else "N/A"
            print(f"    {steps:>6d} adapt steps: {c}/10 laps, mean={t_str}")


if __name__ == "__main__":
    main()
