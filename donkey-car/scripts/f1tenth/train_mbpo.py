"""MBPO training: Model-Based Policy Optimization for RLPP (v4).

v4 fix: separate replay buffers for real and synthetic data.

The root cause of critic explosion in v1-v3 was injecting synthetic
transitions into CrossQ's replay buffer, which poisoned BatchNorm
running statistics. v4 uses MixedReplayBuffer: real data in one buffer,
synthetic in another, with controlled per-batch sampling ratio (default
80% real / 20% synthetic). This keeps BatchNorm dominated by real data.

Also retains v3 stabilizations: adaptive horizon, pessimistic rewards,
reward clamping, loss gate.
"""

import argparse
import os
import time

import numpy as np
import torch
from sbx import CrossQ
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from dynamics_model import DynamicsEnsemble
from mixed_replay_buffer import MixedReplayBuffer
from multi_track_env import MultiTrackEnv
from residual_env import RLPPEnv


class ObsNormalizer:
    """Running mean/std normalizer for observations fed to dynamics model."""

    def __init__(self, obs_dim, clip=10.0):
        self.mean = np.zeros(obs_dim, dtype=np.float64)
        self.var = np.ones(obs_dim, dtype=np.float64)
        self.count = 1e-4
        self.clip = clip

    def update(self, obs):
        batch_mean = obs.mean(axis=0)
        batch_var = obs.var(axis=0)
        batch_count = len(obs)

        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, obs):
        std = np.sqrt(self.var + 1e-8).astype(np.float32)
        return np.clip((obs - self.mean.astype(np.float32)) / std, -self.clip, self.clip)

    def denormalize(self, obs_norm):
        std = np.sqrt(self.var + 1e-8).astype(np.float32)
        return obs_norm * std + self.mean.astype(np.float32)


class DynamicsTrainingCallback(BaseCallback):
    """Periodically train dynamics ensemble and inject synthetic data.

    v3 stabilization:
    - Adaptive horizon starting at 1, increasing when dynamics loss stays low
    - Pessimistic reward penalty proportional to ensemble disagreement
    - Reward clamping to real-data range
    - Synthetic-to-real ratio cap
    """

    def __init__(
        self,
        dynamics: DynamicsEnsemble,
        obs_normalizer: ObsNormalizer,
        train_dynamics_every=2000,
        dynamics_train_epochs=20,
        min_horizon=1,
        max_horizon=5,
        horizon_increase_threshold=12.0,
        horizon_increase_patience=3,
        synthetic_rollouts_per_step=2000,
        warmup_steps=10000,
        loss_gate_threshold=15.0,
        pessimism_beta=1.0,
        reward_clip_range=(-2.0, 5.0),
        max_synthetic_ratio=4.0,
        verbose=1,
    ):
        super().__init__(verbose)
        self.dynamics = dynamics
        self.obs_normalizer = obs_normalizer
        self.train_dynamics_every = train_dynamics_every
        self.dynamics_train_epochs = dynamics_train_epochs

        self.min_horizon = min_horizon
        self.max_horizon = max_horizon
        self.current_horizon = min_horizon
        self.horizon_increase_threshold = horizon_increase_threshold
        self.horizon_increase_patience = horizon_increase_patience
        self._consecutive_good_losses = 0

        self.synthetic_rollouts_per_step = synthetic_rollouts_per_step
        self.warmup_steps = warmup_steps
        self.loss_gate_threshold = loss_gate_threshold
        self.pessimism_beta = pessimism_beta
        self.reward_clip_range = reward_clip_range
        self.max_synthetic_ratio = max_synthetic_ratio

        self._last_dynamics_train = 0
        self._synthetic_transitions_total = 0
        self._dynamics_losses = []
        self._skipped_injections = 0
        self._reward_stats = {"min": 0.0, "max": 1.0, "mean": 0.0}

    def _update_reward_stats(self, rewards):
        self._reward_stats["min"] = float(np.percentile(rewards, 1))
        self._reward_stats["max"] = float(np.percentile(rewards, 99))
        self._reward_stats["mean"] = float(np.mean(rewards))

    def _update_horizon(self, loss):
        if loss < self.horizon_increase_threshold:
            self._consecutive_good_losses += 1
        else:
            self._consecutive_good_losses = 0

        if (self._consecutive_good_losses >= self.horizon_increase_patience
                and self.current_horizon < self.max_horizon):
            self.current_horizon += 1
            self._consecutive_good_losses = 0
            if self.verbose:
                print(f"  [MBPO] Horizon increased to {self.current_horizon}")

    def _on_step(self):
        if self.num_timesteps < self.warmup_steps:
            return True

        if self.num_timesteps - self._last_dynamics_train < self.train_dynamics_every:
            return True

        buffer = self.model.replay_buffer
        n_real = buffer.pos if not buffer.full else buffer.buffer_size
        if n_real < 2000:
            return True

        if self._synthetic_transitions_total > self.max_synthetic_ratio * self.num_timesteps:
            if self.verbose:
                ratio = self._synthetic_transitions_total / max(self.num_timesteps, 1)
                print(f"  [MBPO] SKIPPED injection (ratio {ratio:.1f}x > "
                      f"cap {self.max_synthetic_ratio}x)")
            self._last_dynamics_train = self.num_timesteps
            return True

        self._last_dynamics_train = self.num_timesteps

        n_samples = min(n_real, 50000)
        batch = buffer.sample_real(n_samples) if hasattr(buffer, 'sample_real') else buffer.sample(n_samples)

        obs = batch.observations.cpu().numpy()
        actions = batch.actions.cpu().numpy()
        next_obs = batch.next_observations.cpu().numpy()
        rewards = batch.rewards.cpu().numpy().flatten()
        dones = batch.dones.cpu().numpy().flatten()

        self._update_reward_stats(rewards)
        self.obs_normalizer.update(obs)

        obs_norm = self.obs_normalizer.normalize(obs)
        next_obs_norm = self.obs_normalizer.normalize(next_obs)

        losses = self.dynamics.train_on_batch(
            obs_norm, actions, next_obs_norm, rewards, dones,
            n_epochs=self.dynamics_train_epochs,
        )

        final_loss = losses[-1]
        self._dynamics_losses.append(final_loss)
        self._update_horizon(final_loss)

        if self.verbose:
            print(f"  [MBPO] Dynamics loss: {final_loss:.4f} "
                  f"(trained on {n_samples}, step {self.num_timesteps}, "
                  f"horizon={self.current_horizon})")

        if final_loss > self.loss_gate_threshold:
            self._skipped_injections += 1
            if self.verbose:
                print(f"  [MBPO] SKIPPED injection (loss {final_loss:.1f} > "
                      f"threshold {self.loss_gate_threshold}), "
                      f"skipped {self._skipped_injections} total")
            return True

        start_indices = np.random.choice(
            len(obs), min(self.synthetic_rollouts_per_step, len(obs)),
            replace=False,
        )
        start_obs_norm = obs_norm[start_indices]

        synthetic = self._generate_pessimistic_rollouts(
            start_obs_norm, self.current_horizon,
        )

        n_synthetic = len(synthetic["obs"])
        self._synthetic_transitions_total += n_synthetic

        for i in range(n_synthetic):
            buffer.add_synthetic(
                obs=synthetic["obs"][i:i+1],
                next_obs=synthetic["next_obs"][i:i+1],
                action=synthetic["action"][i:i+1],
                reward=synthetic["reward"][i:i+1].reshape(1),
                done=synthetic["done"][i:i+1].reshape(1),
                infos=[{}],
            )

        if self.verbose:
            ratio = self._synthetic_transitions_total / max(self.num_timesteps, 1)
            mean_disagree = np.mean(synthetic.get("disagreement", [0.0]))
            syn_buf_size = buffer.synthetic_size if hasattr(buffer, 'synthetic_size') else '?'
            print(f"  [MBPO] Added {n_synthetic} synthetic transitions "
                  f"(total: {self._synthetic_transitions_total}, "
                  f"ratio: {ratio:.1f}x, "
                  f"mean_disagree: {mean_disagree:.4f}, "
                  f"syn_buf: {syn_buf_size})")

            if hasattr(self.model, 'logger') and self.model.logger is not None:
                try:
                    logs = self.model.logger.name_to_value
                    critic_loss = logs.get('train/critic_loss', None)
                    if critic_loss is not None:
                        print(f"  [MBPO] Current critic_loss: {critic_loss:.4f}")
                        if critic_loss > 1e6:
                            print(f"  [MBPO] WARNING: critic_loss > 1M, possible explosion!")
                except Exception:
                    pass

        return True

    def _generate_pessimistic_rollouts(self, start_obs_norm, horizon):
        """Generate rollouts with pessimistic reward penalty and clamping."""
        all_obs, all_act, all_next, all_rew, all_done, all_disagree = \
            [], [], [], [], [], []

        r_min, r_max = self.reward_clip_range

        obs_norm = start_obs_norm.copy()
        for step in range(horizon):
            obs_raw = self.obs_normalizer.denormalize(obs_norm)
            action, _ = self.model.predict(obs_raw, deterministic=False)

            next_obs_norm, reward, done, disagreement = \
                self.dynamics.predict_with_disagreement(obs_norm, action)

            reward_penalized = reward - self.pessimism_beta * disagreement
            reward_clamped = np.clip(reward_penalized, r_min, r_max)

            next_obs_raw = self.obs_normalizer.denormalize(next_obs_norm)

            all_obs.append(obs_raw)
            all_act.append(action)
            all_next.append(next_obs_raw)
            all_rew.append(reward_clamped)
            all_done.append(done)
            all_disagree.append(disagreement)

            alive = ~done
            if not alive.any():
                break
            obs_norm = next_obs_norm
            if done.any():
                replace_idx = np.random.choice(
                    len(start_obs_norm), done.sum(), replace=True,
                )
                obs_norm[done] = start_obs_norm[replace_idx]

        return {
            "obs": np.concatenate(all_obs, axis=0),
            "action": np.concatenate(all_act, axis=0),
            "next_obs": np.concatenate(all_next, axis=0),
            "reward": np.concatenate(all_rew, axis=0),
            "done": np.concatenate(all_done, axis=0),
            "disagreement": np.concatenate(all_disagree, axis=0),
        }


def make_multi_track_env(
    real_tracks, tracks_dir, generated_tracks_dir,
    n_generated, prob_generated, seed=None, **env_kwargs
):
    def _init():
        env = MultiTrackEnv(
            real_tracks=real_tracks,
            tracks_dir=tracks_dir,
            generated_tracks_dir=generated_tracks_dir,
            n_generated=n_generated,
            prob_generated=prob_generated,
            pregenerate=True,
            **env_kwargs,
        )
        if seed is not None:
            env.reset(seed=seed)
        return Monitor(env)
    return _init


def make_eval_env(track_name, tracks_dir, seed=None, **env_kwargs):
    def _init():
        kw = dict(env_kwargs)
        kw["mu_noise_std"] = 0.0
        kw["velocity_curriculum"] = False
        env = RLPPEnv(track_name=track_name, tracks_dir=tracks_dir, **kw)
        if seed is not None:
            env.reset(seed=seed)
        return Monitor(env)
    return _init


def eval_on_track(model, track_name, tracks_dir=None, alpha_rl=0.55,
                  n_laps=10):
    """Run evaluation on a single track, return lap times."""
    if tracks_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tracks_dir = os.path.join(script_dir, "f1tenth_racetracks")

    eval_env = RLPPEnv(
        track_name=track_name, tracks_dir=tracks_dir,
        alpha_rl=alpha_rl, velocity_gain=0.75,
        mu_noise_std=0.0, velocity_curriculum=False, max_laps=10,
    )

    lap_times = []
    for lap_i in range(n_laps):
        obs, _ = eval_env.reset()
        eval_env.alpha_rl = alpha_rl
        steps = 0
        cumul_s = 0.0
        prev_s = eval_env.raceline_s[eval_env._closest_idx]
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            steps += 1
            cur_s = eval_env.raceline_s[eval_env._closest_idx]
            ds = cur_s - prev_s
            if ds < -eval_env.total_s / 2: ds += eval_env.total_s
            elif ds > eval_env.total_s / 2: ds -= eval_env.total_s
            cumul_s += ds
            prev_s = cur_s
            if cumul_s >= eval_env.total_s and steps > 100:
                lap_times.append(steps * eval_env.controller_dt)
                print(f"  Lap {lap_i+1}: {lap_times[-1]:.2f}s")
                break
            if terminated or truncated or steps > 100000:
                print(f"  Lap {lap_i+1}: DNF (steps={steps})")
                break

    return lap_times


def main():
    parser = argparse.ArgumentParser(description="Train MBPO + CrossQ + Multi-Track (v4 — separate replay buffers)")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--utd", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)

    parser.add_argument("--n-generated-tracks", type=int, default=50)
    parser.add_argument("--prob-generated", type=float, default=0.5)
    parser.add_argument("--eval-track", default="Spielberg")
    parser.add_argument("--eval-tracks", nargs="*", default=None,
                        help="Additional tracks for zero-shot eval")

    parser.add_argument("--dynamics-train-every", type=int, default=2000)
    parser.add_argument("--dynamics-epochs", type=int, default=20)
    parser.add_argument("--min-horizon", type=int, default=1)
    parser.add_argument("--max-horizon", type=int, default=3)
    parser.add_argument("--horizon-threshold", type=float, default=12.0,
                        help="Dynamics loss threshold for horizon increase")
    parser.add_argument("--horizon-patience", type=int, default=3,
                        help="Consecutive good updates before horizon increase")
    parser.add_argument("--synthetic-rollouts", type=int, default=2000,
                        help="Rollouts per dynamics update")
    parser.add_argument("--warmup-steps", type=int, default=10000,
                        help="Pure CrossQ steps before activating MBPO")
    parser.add_argument("--loss-gate", type=float, default=15.0,
                        help="Skip synthetic injection if dynamics loss exceeds this")
    parser.add_argument("--pessimism-beta", type=float, default=1.0,
                        help="Penalty coefficient on ensemble disagreement")
    parser.add_argument("--max-synthetic-ratio", type=float, default=4.0,
                        help="Max synthetic-to-real ratio in buffer")
    parser.add_argument("--reward-clip-min", type=float, default=-2.0)
    parser.add_argument("--reward-clip-max", type=float, default=5.0)
    parser.add_argument("--dynamics-device", default="cpu",
                        help="Device for dynamics model (cpu to avoid GPU OOM)")

    parser.add_argument("--real-fraction", type=float, default=0.8,
                        help="Fraction of each batch sampled from real data (rest from synthetic)")
    parser.add_argument("--synthetic-buffer-size", type=int, default=500_000,
                        help="Size of the separate synthetic replay buffer")
    parser.add_argument("--tag", default="mbpo_v4")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"=== MBPO v4 Training: {args.tag} ===")
    print(f"  Real steps: {args.steps}")
    print(f"  n_envs={args.n_envs}, UTD={args.utd}, lr={args.lr}")
    print(f"  Tracks: {args.n_generated_tracks} generated + real F1TENTH")
    print(f"  Dynamics: train every {args.dynamics_train_every} steps, "
          f"rollouts={args.synthetic_rollouts}")
    print(f"  Horizon: {args.min_horizon} -> {args.max_horizon} "
          f"(threshold={args.horizon_threshold}, patience={args.horizon_patience})")
    print(f"  Warmup: {args.warmup_steps} pure CrossQ steps")
    print(f"  Loss gate: {args.loss_gate}")
    print(f"  Pessimism beta: {args.pessimism_beta}")
    print(f"  Max synthetic ratio: {args.max_synthetic_ratio}x")
    print(f"  Reward clamp: [{args.reward_clip_min}, {args.reward_clip_max}]")
    print(f"  Real fraction: {args.real_fraction} (synthetic: {1-args.real_fraction:.1f})")
    print(f"  Synthetic buffer size: {args.synthetic_buffer_size}")
    print(f"  Dynamics device: {args.dynamics_device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    tracks_dir = os.path.join(script_dir, "f1tenth_racetracks")
    gen_tracks_dir = os.path.join(script_dir, "generated_tracks")

    env_kwargs = dict(
        alpha_rl=1.0,
        velocity_gain=0.75,
        mu_noise_std=0.15,
        velocity_curriculum=True,
    )

    print("Creating multi-track environments...")
    if args.n_envs > 1:
        train_env = SubprocVecEnv([
            make_multi_track_env(
                real_tracks=None, tracks_dir=tracks_dir,
                generated_tracks_dir=gen_tracks_dir,
                n_generated=args.n_generated_tracks,
                prob_generated=args.prob_generated,
                seed=args.seed + i, **env_kwargs,
            )
            for i in range(args.n_envs)
        ])
    else:
        train_env = DummyVecEnv([
            make_multi_track_env(
                real_tracks=None, tracks_dir=tracks_dir,
                generated_tracks_dir=gen_tracks_dir,
                n_generated=args.n_generated_tracks,
                prob_generated=args.prob_generated,
                seed=args.seed, **env_kwargs,
            )
        ])

    eval_env = DummyVecEnv([
        make_eval_env(args.eval_track, tracks_dir, seed=args.seed + 100)
    ])

    print("Initializing CrossQ policy with MixedReplayBuffer...")
    model = CrossQ(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=args.lr,
        buffer_size=2_000_000,
        batch_size=args.batch_size,
        gamma=0.99,
        gradient_steps=args.utd,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        seed=args.seed,
        tensorboard_log=f"./logs/mbpo/",
    )
    mixed_buffer = MixedReplayBuffer(
        buffer_size=2_000_000,
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        device="cpu",
        n_envs=args.n_envs,
        synthetic_buffer_size=args.synthetic_buffer_size,
        real_fraction=args.real_fraction,
    )
    model.replay_buffer = mixed_buffer

    obs_dim = train_env.observation_space.shape[0]
    act_dim = train_env.action_space.shape[0]
    dyn_device = args.dynamics_device
    print(f"Initializing dynamics ensemble ({obs_dim}D obs, {act_dim}D act, "
          f"device={dyn_device})...")
    dynamics = DynamicsEnsemble(
        obs_dim=obs_dim,
        act_dim=act_dim,
        n_models=5,
        hidden_dims=(256, 256, 256),
        lr=1e-3,
        device=dyn_device,
    )

    obs_normalizer = ObsNormalizer(obs_dim)

    save_dir = f"./mbpo/{args.tag}"
    dynamics_cb = DynamicsTrainingCallback(
        dynamics=dynamics,
        obs_normalizer=obs_normalizer,
        train_dynamics_every=args.dynamics_train_every,
        dynamics_train_epochs=args.dynamics_epochs,
        min_horizon=args.min_horizon,
        max_horizon=args.max_horizon,
        horizon_increase_threshold=args.horizon_threshold,
        horizon_increase_patience=args.horizon_patience,
        synthetic_rollouts_per_step=args.synthetic_rollouts,
        warmup_steps=args.warmup_steps,
        loss_gate_threshold=args.loss_gate,
        pessimism_beta=args.pessimism_beta,
        reward_clip_range=(args.reward_clip_min, args.reward_clip_max),
        max_synthetic_ratio=args.max_synthetic_ratio,
    )

    callbacks = CallbackList([
        dynamics_cb,
        CheckpointCallback(
            save_freq=10_000,
            save_path=f"{save_dir}/checkpoints/",
            name_prefix="mbpo",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=f"{save_dir}/best_model/",
            eval_freq=5_000,
            n_eval_episodes=5,
            deterministic=True,
        ),
    ])

    print(f"\n{'='*60}")
    print(f"Starting MBPO v4 training ({args.steps} real steps)...")
    print(f"  MixedReplayBuffer: {args.real_fraction*100:.0f}% real / "
          f"{(1-args.real_fraction)*100:.0f}% synthetic per batch")
    print(f"{'='*60}\n")

    t0 = time.time()
    model.learn(
        total_timesteps=args.steps,
        callback=callbacks,
        progress_bar=True,
        tb_log_name=args.tag,
    )
    wall_time = time.time() - t0

    model.save(f"{save_dir}/final_model")
    dynamics.save(f"{save_dir}/dynamics_ensemble.pt")

    print(f"\n=== {args.tag} complete ===")
    print(f"  Wall clock: {wall_time:.1f}s ({wall_time/60:.1f}m)")
    print(f"  Real steps: {args.steps}")
    print(f"  Synthetic transitions: {dynamics_cb._synthetic_transitions_total}")
    amp = dynamics_cb._synthetic_transitions_total / max(args.steps, 1)
    print(f"  Amplification: {amp:.1f}x")
    print(f"  Final horizon: {dynamics_cb.current_horizon}")
    print(f"  Skipped injections: {dynamics_cb._skipped_injections}")
    print(f"  Effective FPS: {args.steps / wall_time:.0f}")
    if dynamics_cb._dynamics_losses:
        print(f"  Final dynamics loss: {dynamics_cb._dynamics_losses[-1]:.4f}")
        print(f"  Best dynamics loss: {min(dynamics_cb._dynamics_losses):.4f}")

    # Eval on primary track
    eval_tracks = [args.eval_track]
    if args.eval_tracks:
        eval_tracks.extend(args.eval_tracks)

    best_path = f"{save_dir}/best_model/best_model.zip"
    if os.path.exists(best_path):
        best_model = CrossQ.load(best_path)
        print("Using best model checkpoint")
    else:
        best_model = model
        print("Using final model")

    for track in eval_tracks:
        print(f"\n=== Evaluation on {track} ===")
        lap_times = eval_on_track(best_model, track, tracks_dir, alpha_rl=0.55)
        if lap_times:
            print(f"\n  {len(lap_times)}/10 completed")
            print(f"  Mean: {np.mean(lap_times):.2f} +/- {np.std(lap_times):.3f}s")
            print(f"  Best: {np.min(lap_times):.2f}s")
        else:
            print("\n  0/10 completed")


if __name__ == "__main__":
    main()
