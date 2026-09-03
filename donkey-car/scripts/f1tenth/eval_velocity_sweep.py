"""Sweep velocity_gain (alpha_v) x alpha_rl on v3_best to test PP over-braking.
Adaptation and eval use MATCHED velocity_gain so the residual policy sees the
same base controller it is evaluated against."""
import os, sys, gc, pickle, time, argparse
import numpy as np
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax, jax.numpy as jnp
from stable_baselines3.common.vec_env import DummyVecEnv

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
from residual_env import RLPPEnv
from train_reptile import create_model, snapshot_params, load_params, make_env_fn

HELD_OUT = ["Austin", "Monza", "Silverstone"]
VELOCITY_GAINS = [0.3, 0.5, 0.7, 0.9]
ALPHA_VALUES = [0.55, 0.70]
ADAPT = 4_000
N_LAPS = 10

def evaluate_track(model, track_name, tracks_dir, n_laps, alpha_rl, velocity_gain, action_scaling):
    env = RLPPEnv(
        track_name=track_name, tracks_dir=tracks_dir,
        alpha_rl=alpha_rl, velocity_gain=velocity_gain,
        mu_noise_std=0.0, velocity_curriculum=False, max_laps=n_laps,
        action_scaling=action_scaling,
    )
    real_rl_xy = np.ascontiguousarray(env.raceline[:, 1:3])
    real_rl_psi = env.raceline[:, 3]
    lap_times, all_rl_devs, top_speeds = [], [], []

    for _ in range(n_laps):
        obs, _ = env.reset()
        env.alpha_rl = alpha_rl
        steps = 0
        cumul_s = 0.0
        prev_s = env.raceline_s[env._closest_idx]
        vmax_seen = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            px, py = env._last_pos[0], env._last_pos[1]
            vx_now = float(abs(env.sim.agents[0].state[3]))
            vmax_seen = max(vmax_seen, vx_now)
            rl_idx = int(np.argmin((real_rl_xy[:, 0] - px)**2 + (real_rl_xy[:, 1] - py)**2))
            ref_pos = real_rl_xy[rl_idx]
            ref_psi = real_rl_psi[rl_idx]
            ev = np.array([px - ref_pos[0], py - ref_pos[1]])
            d_rl = -np.sin(ref_psi) * ev[0] + np.cos(ref_psi) * ev[1]
            all_rl_devs.append(abs(d_rl))
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
                top_speeds.append(vmax_seen)
                break
            if terminated or truncated or steps > 20000:
                break
    env.close()
    completed = len(lap_times)
    mean_t = float(np.mean(lap_times)) if lap_times else None
    mean_rl = float(np.mean(all_rl_devs)) if all_rl_devs else None
    mean_vtop = float(np.mean(top_speeds)) if top_speeds else None
    return completed, mean_t, mean_rl, mean_vtop

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--utd", type=int, default=20)
    parser.add_argument("--inner-lr", type=float, default=1e-3)
    parser.add_argument("--inner-steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    tracks_dir = os.path.join(_SCRIPT_DIR, "f1tenth_racetracks")
    ckpt_path = os.path.join(_SCRIPT_DIR, "reptile/reptile_v3/meta_params_best.pkl")
    action_scaling = (0.05, 1.0)

    with open(ckpt_path, "rb") as f:
        meta_params_np = pickle.load(f)
    meta_params = jax.tree.map(jnp.array, meta_params_np)

    print(f"{'Gain':<5} {'Alpha':<6} {'Track':<12} {'Laps':>6} {'Time':>8} {'|d_rl|':>8} {'Vtop':>6} {'Wall':>6}")
    print("-" * 74)
    sys.stdout.flush()

    for track in HELD_OUT:
        for gain in VELOCITY_GAINS:
            # Adapt with matched velocity_gain
            env_kwargs = dict(alpha_rl=1.0, velocity_gain=gain,
                              mu_noise_std=0.15, velocity_curriculum=True,
                              action_scaling=action_scaling)
            env = DummyVecEnv([make_env_fn(track, tracks_dir, seed=args.seed+200, **env_kwargs)])
            model = create_model(env, args)
            load_params(model, meta_params)
            model.learn(total_timesteps=ADAPT)
            adapted = snapshot_params(model)
            env.close()
            del model
            gc.collect()

            for alpha in ALPHA_VALUES:
                t0 = time.time()
                eval_env = DummyVecEnv([make_env_fn(track, tracks_dir, seed=args.seed+300,
                    alpha_rl=alpha, velocity_gain=gain, mu_noise_std=0.0,
                    velocity_curriculum=False, action_scaling=action_scaling)])
                eval_model = create_model(eval_env, args)
                load_params(eval_model, adapted)
                eval_env.close()

                comp, mean_t, mean_rl, vtop = evaluate_track(
                    eval_model, track, tracks_dir, N_LAPS, alpha, gain, action_scaling)
                wall = time.time() - t0

                t_str = f"{mean_t:.2f}s" if mean_t else "DNF"
                rl_str = f"{mean_rl:.4f}m" if mean_rl else "N/A"
                v_str = f"{vtop:.1f}" if vtop else "N/A"
                print(f"{gain:<5.1f} {alpha:<6.2f} {track:<12} {comp:>2}/{N_LAPS:<2}  {t_str:>8} {rl_str:>8} {v_str:>6} {wall:>5.0f}s")
                sys.stdout.flush()

                del eval_model
                gc.collect()

if __name__ == "__main__":
    main()
