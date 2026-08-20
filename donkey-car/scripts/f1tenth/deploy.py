"""Evaluate a trained RLPP model or run the PP baseline.

Usage:
  python deploy.py --track Spielberg --model best_model/Spielberg/best_model.zip
  python deploy.py --track Spielberg --baseline
"""

import argparse
import numpy as np

from stable_baselines3 import SAC
from residual_env import RLPPEnv


def run_laps(env, model, n_laps=10, alpha_rl=0.55, deterministic=True):
    lap_times = []
    lateral_devs = []
    completed = 0

    for lap in range(n_laps):
        obs, _ = env.reset()
        env.alpha_rl = alpha_rl
        start_s = env.raceline_s[env._closest_idx]
        steps = 0
        lap_devs = []
        cumul_s = 0.0
        prev_s = start_s

        while True:
            if model is not None:
                action, _ = model.predict(obs, deterministic=deterministic)
            else:
                action = np.zeros(2, dtype=np.float32)

            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            lap_devs.append(abs(obs[0]))

            cur_s = env.raceline_s[env._closest_idx]
            ds = cur_s - prev_s
            if ds < -env.total_s / 2:
                ds += env.total_s
            elif ds > env.total_s / 2:
                ds -= env.total_s
            cumul_s += ds
            prev_s = cur_s

            if cumul_s >= env.total_s and steps > 100:
                lap_time = steps * env.controller_dt
                lap_times.append(lap_time)
                lateral_devs.append(np.mean(lap_devs))
                completed += 1
                print(f"  Lap {lap+1}: {lap_time:.3f}s, avg|d|={np.mean(lap_devs):.3f}m")
                break

            if terminated or truncated or steps > 100000:
                print(f"  Lap {lap+1}: DNF (collision={info.get('collision', False)}, steps={steps})")
                break

    if lap_times:
        print(f"\n  {completed}/{n_laps} laps completed")
        print(f"  Mean: {np.mean(lap_times):.3f} +/- {np.std(lap_times):.3f}s")
        print(f"  Best: {np.min(lap_times):.3f}s")
        print(f"  Mean |d|: {np.mean(lateral_devs):.3f}m")
    return lap_times, lateral_devs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", default="Spielberg")
    parser.add_argument("--model", default=None)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--alpha-rl", type=float, default=0.55)
    parser.add_argument("--n-laps", type=int, default=10)
    args = parser.parse_args()

    env = RLPPEnv(
        track_name=args.track,
        alpha_rl=args.alpha_rl,
        velocity_gain=0.75,
        mu_noise_std=0.0,
        velocity_curriculum=False,
        max_laps=10,
    )

    if args.baseline:
        print("=== PP Baseline ===")
        run_laps(env, None, n_laps=args.n_laps, alpha_rl=0.0)
    elif args.model:
        model = SAC.load(args.model)
        print(f"=== RLPP (alpha_rl={args.alpha_rl}) ===")
        run_laps(env, model, n_laps=args.n_laps, alpha_rl=args.alpha_rl)
    else:
        parser.error("Specify --model or --baseline")


if __name__ == "__main__":
    main()
