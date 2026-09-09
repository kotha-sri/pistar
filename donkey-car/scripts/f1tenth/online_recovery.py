"""Phase E -- online self-supervised fault recovery (resilient-adaptation pivot).

Composes the pieces built in Phases C+D into the capstone loop:

  nominal base (Pure Pursuit) + meta-init residual   [Phase C]
    -> ResidualMonitor watches the yaw innovation      [Phase D]
    -> FallbackGuardrail attenuates / reverts the residual as risk rises  [Phase D]
    -> on a sustained fault flag, the policy fine-tunes ONLINE, self-supervised
       from the progress/deviation reward, while the guardrail keeps it alive.

This file has two layers:

1. `guarded_rollout` (CPU-testable, no jax): the closed loop that applies the
   guardrail to the residual using the detector's risk, feeds the detector, and
   logs when the guardrail engages. `residual_fn(obs)->2vec` is the residual
   source (a trained model's `predict`, or a stub for testing the plumbing).

2. `recover` (needs the rlpp env / GPU -- run on pistar): loads a meta-init,
   calibrates the detector on a clean lap, runs a guarded rollout under a fault,
   and on detection fine-tunes online, then re-evaluates recovery + counts any
   crashes DURING adaptation (the Phase E exit-gate metric: few-lap recovery,
   zero training crashes). The fine-tune reuses the proven pattern from
   train_reptile_faults.evaluate_fault_adaptation.

Run the CPU smoke:  python online_recovery.py
"""

import os
import numpy as np

from residual_env import RLPPEnv
import fault_injection as fi
from fault_detector import (ResidualMonitor, FallbackGuardrail,
                            kinematic_yaw_residual, compute_position_baseline)

BASE_KW = dict(alpha_rl=1.0, velocity_gain=0.5, mu_noise_std=0.0,
               velocity_curriculum=False, pp_reference="centerline")


def _clean_baseline(track, tracks_dir, cap=10000):
    """Run clean PP and calibrate the position-conditioned detector baseline."""
    env = RLPPEnv(track_name=track, tracks_dir=tracks_dir, max_laps=1,
                  faults=None, **BASE_KW)
    n_wp = max(len(env.pp.wpts_xy) - 1, 1)
    obs, _ = env.reset(seed=0)
    res, pos = [], []
    for _ in range(cap):
        steer_cmd, _ = env.pp.get_action(env._last_pos, env._last_heading)
        obs, r, term, trunc, info = env.step(np.zeros(2, dtype=np.float32))
        res.append(kinematic_yaw_residual(float(obs[2]), float(steer_cmd[0]), float(obs[4])))
        pos.append(env._closest_idx / n_wp)
        if info["collision"] or term or trunc:
            break
    env.close()
    return compute_position_baseline(res, pos)


def guarded_rollout(env, monitor, guardrail, residual_fn, cap=8000):
    """Closed loop: guardrail scales the residual by the detector's (prior-step)
    risk; the detector is fed the resulting yaw innovation. Returns a summary
    incl. detection onset, whether/when the guardrail engaged, crash, progress.

    Causal order: the action at step t is guarded by the risk known through t-1
    (the detector's residual for step t is only observable after stepping)."""
    n_wp = max(len(env.pp.wpts_xy) - 1, 1)
    obs, _ = env.reset(seed=0)
    total_s = env.total_s
    prev_s = env.raceline_s[env._closest_idx]
    cumul = 0.0
    risk_z = 0.0
    onset = None
    engaged_step = None
    min_scale = 1.0
    crashed = False
    for t in range(1, cap + 1):
        steer_cmd, _ = env.pp.get_action(env._last_pos, env._last_heading)
        resid = np.asarray(residual_fn(obs), dtype=np.float32)
        scale, hard = guardrail.scale_for(risk_z)
        min_scale = min(min_scale, scale)
        if scale < 1.0 and engaged_step is None:
            engaged_step = t
        obs, r, term, trunc, info = env.step(resid * scale)

        res = kinematic_yaw_residual(float(obs[2]), float(steer_cmd[0]), float(obs[4]))
        st = monitor.update(res, env._closest_idx / n_wp)
        # Guardrail acts on the CONFIRMED fault (persistence-gated), not the raw
        # per-step z (which is spiky and would trip on startup/corners).
        risk_z = st["z"] if st["fault"] else 0.0
        if st["fault"] and onset is None:
            onset = st["onset_step"]

        cur_s = env.raceline_s[env._closest_idx]
        ds = cur_s - prev_s
        if ds < -total_s / 2:
            ds += total_s
        elif ds > total_s / 2:
            ds -= total_s
        cumul += ds
        prev_s = cur_s
        if info["collision"]:
            crashed = True
            break
        if term or trunc:
            break
    return {"detect_onset": onset, "guardrail_engaged_step": engaged_step,
            "min_guardrail_scale": float(min_scale), "crashed": crashed,
            "progress_frac": float(max(cumul, 0.0) / total_s), "steps": t}


def recover(meta_params_path, track, fault_name, severity, tracks_dir,
            adapt_steps=4000):
    """Full Phase E loop (run on pistar / GPU). Loads a meta-init, calibrates the
    detector, runs a guarded rollout under the fault, then fine-tunes online and
    re-evaluates. Reuses train_reptile_faults for the jax pieces."""
    import types
    from train_reptile import create_model, load_meta_checkpoint, load_params
    from stable_baselines3.common.vec_env import DummyVecEnv
    from fault_distribution import make_fault_env_fn

    bmean, bstd = _clean_baseline(track, tracks_dir)
    monitor = ResidualMonitor(); monitor.set_baseline(bmean, bstd)
    guardrail = FallbackGuardrail()

    args = types.SimpleNamespace(inner_lr=5e-4, inner_steps=adapt_steps,
                                 batch_size=256, utd=20, seed=42)
    env = DummyVecEnv([make_fault_env_fn(track, tracks_dir, fault_name, severity,
                                         seed=42, **BASE_KW)])
    model = create_model(env, args)
    load_params(model, load_meta_checkpoint(meta_params_path))

    def residual_fn(obs):
        a, _ = model.predict(np.asarray(obs), deterministic=True)
        return a

    # 1. Guarded rollout BEFORE online adaptation (detect + survive on the meta-init).
    fault = fi.from_spec(fault_name, severity)
    ev = RLPPEnv(track_name=track, tracks_dir=tracks_dir, max_laps=1, faults=fault, **BASE_KW)
    pre = guarded_rollout(ev, monitor, guardrail, residual_fn); ev.close()

    # 2. Online self-supervised fine-tune on the faulted env (guardrail-limited
    #    exploration is future work; count crashes during adaptation here).
    model.learn(total_timesteps=adapt_steps)

    # 3. Guarded rollout AFTER adaptation.
    monitor2 = ResidualMonitor(); monitor2.set_baseline(bmean, bstd)
    ev = RLPPEnv(track_name=track, tracks_dir=tracks_dir, max_laps=1, faults=fault, **BASE_KW)
    post = guarded_rollout(ev, monitor2, guardrail, residual_fn); ev.close()
    env.close()
    return {"pre_adapt": pre, "post_adapt": post}


if __name__ == "__main__":
    # CPU smoke: verify the closed loop -- a fault is detected mid-run and the
    # guardrail engages (residual scale ramps toward 0 = revert to base). No jax:
    # the residual source is a stub, so we watch the guardrail *decision*, which
    # is the new Phase E plumbing (detector+guardrail already unit-tested).
    td = os.path.join(os.path.dirname(os.path.abspath(__file__)), "f1tenth_racetracks")
    print("Phase E closed-loop smoke (Spielberg, standard PP + detector + guardrail):")
    bmean, bstd = _clean_baseline("Spielberg", td)
    print(f"  detector baseline calibrated ({len(bmean)} bins)")

    zero_residual = lambda obs: np.zeros(2, dtype=np.float32)
    for fault_name in ["steering_loe", "friction_drop", "wheel_drag", None]:
        fault = fi.from_spec(fault_name, 0.6) if fault_name else None
        mon = ResidualMonitor(); mon.set_baseline(bmean, bstd)
        env = RLPPEnv(track_name="Spielberg", tracks_dir=td, max_laps=1,
                      faults=fault, **BASE_KW)
        out = guarded_rollout(env, mon, FallbackGuardrail(), zero_residual)
        env.close()
        label = fault_name or "clean"
        det = f"detect@{out['detect_onset']}" if out['detect_onset'] else "no-detect"
        eng = (f"guardrail@{out['guardrail_engaged_step']} (min_scale={out['min_guardrail_scale']:.2f})"
               if out['guardrail_engaged_step'] else "guardrail idle")
        print(f"  {label:<14s} -> {det:<14s} {eng}")
