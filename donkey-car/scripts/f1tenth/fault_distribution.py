"""Fault task distribution for Phase C meta-adaptation (Reptile over faults).

A "task" for the fault-Reptile trainer is a (track, fault_type, severity) triple.
The meta-init is trained to adapt quickly to any such task, then evaluated on:
  - held-out SEVERITIES of trained fault types  (in-distribution generalization)
  - held-out fault TYPES it never trained on     (out-of-distribution / the headline)

Analog of `multi_track_env.py` (which sampled tracks); here we sample faults.
"""

import fault_injection as fi

# Fault TYPES the meta-init trains on (severity sampled per task).
# Includes obs_noise + steering_bias: unlike a pure PP baseline (Phase B showed
# them inert), the RL residual READS the observation and can be pushed off-line,
# so these are learnable faults for the residual.
TRAIN_FAULTS = [
    "friction_drop", "low_grip_patch", "tire_stiffness",
    "steering_loe", "actuator_latency", "wheel_drag",
    "obs_noise", "steering_bias",
]

# Never-seen fault TYPES held out for the OOD generalization number (the headline
# metric existing work doesn't report). Deliberately span both classes.
HELD_OUT_FAULTS = ["friction_drop_mid", "obs_latency", "mass_change"]

# Severity band: "recoverable but challenging." Phase B breakdown severities for a
# NON-adaptive PP were ~0.5-0.75; the residual should extend that, so we train in a
# moderate band that gives a learnable signal without being hopeless.
SEV_RANGE = (0.2, 0.6)


def sample_task(rng, tracks, fault_names=TRAIN_FAULTS, sev_range=SEV_RANGE):
    """Sample one (track, fault_name, severity) task."""
    track = tracks[rng.randint(len(tracks))]
    name = fault_names[rng.randint(len(fault_names))]
    sev = float(rng.uniform(*sev_range))
    return track, name, sev


def make_fault_env_fn(track_name, tracks_dir, fault_name, severity, seed=None,
                      **env_kwargs):
    """Env factory that pins one fault (task) for a whole inner loop / eval."""
    from residual_env import RLPPEnv
    from stable_baselines3.common.monitor import Monitor

    def _init():
        fault = fi.from_spec(fault_name, severity) if fault_name is not None else None
        env = RLPPEnv(track_name=track_name, tracks_dir=tracks_dir, faults=fault,
                      **env_kwargs)
        if seed is not None:
            env.reset(seed=seed)
        return Monitor(env)

    return _init
