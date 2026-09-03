"""Sweep the centerline->raceline blend for PP ALONE (no RL).
Find the tightest reference PP can still drive without crashing.
blend=0 -> centerline, blend=1 -> full feasible raceline.
"""
import numpy as np
from residual_env import RLPPEnv

TRACKS = ["Austin", "Monza", "Silverstone", "Spielberg"]
BLENDS = [0.0, 0.25, 0.5, 0.75, 1.0]
N_LAPS = 2
MAX_STEPS = 20000


def d_rl(env, px, py):
    i = int(np.argmin((env.real_rl_xy[:, 0] - px) ** 2
                      + (env.real_rl_xy[:, 1] - py) ** 2))
    rp = env.real_rl_xy[i]; rpsi = env.real_rl_psi[i]
    return abs(-np.sin(rpsi) * (px - rp[0]) + np.cos(rpsi) * (py - rp[1]))


def run(track, blend):
    ref = "centerline" if blend == 0.0 else "raceline"
    env = RLPPEnv(track_name=track, pp_reference=ref, pp_blend=blend,
                  alpha_rl=1.0, velocity_gain=0.5, mu_noise_std=0.0,
                  velocity_curriculum=False, action_scaling=(0.05, 1.0),
                  max_laps=N_LAPS)
    completed = 0; devs = []
    zero = np.zeros(2, dtype=np.float32)
    for _ in range(N_LAPS):
        obs, _ = env.reset()
        steps = 0; cumul_s = 0.0
        prev_s = env.raceline_s[env._closest_idx]
        while True:
            obs, r, term, trunc, info = env.step(zero)
            steps += 1
            devs.append(d_rl(env, env._last_pos[0], env._last_pos[1]))
            cur_s = env.raceline_s[env._closest_idx]
            ds = cur_s - prev_s
            if ds < -env.total_s / 2: ds += env.total_s
            elif ds > env.total_s / 2: ds -= env.total_s
            cumul_s += ds; prev_s = cur_s
            if cumul_s >= env.total_s and steps > 100:
                completed += 1; break
            if term or trunc or steps > MAX_STEPS: break
    env.close()
    return completed, (float(np.mean(devs)) if devs else None)


def main():
    print("=== PP-alone blend sweep (laps / mean|d_rl|) ===")
    hdr = "track       " + "".join(f"blend={b:<12.2f}" for b in BLENDS)
    print(hdr)
    for track in TRACKS:
        row = f"{track:<12s}"
        for b in BLENDS:
            c, dev = run(track, b)
            row += f"{c}/{N_LAPS} {dev:.3f}     "
        print(row, flush=True)


if __name__ == "__main__":
    main()
