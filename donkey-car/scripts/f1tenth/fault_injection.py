"""Fault / degradation injection for the F1TENTH residual-RL environment.

Phase 0 of the resilient-adaptation pivot (see fault_adaptation_plan.md):
prove the F1TENTH sim can inject realistic faults through hooks that ALREADY
exist in the vendored simulator, split into two classes:

  IN-MODEL faults      -- expressible by perturbing VEHICLE_PARAMS (grip, tire
                          stiffness, mass). Applied via Simulator.update_params.
                          This is where classical adaptive control / on-track
                          system-ID should win; they serve as the honest control.

  OUT-OF-MODEL faults  -- NOT representable by the vehicle model's parameters
                          (steering trim bias, actuator loss-of-effectiveness,
                          actuator latency, sensor latency/noise). Applied at the
                          command / observation layer. This is the frontier where
                          a model-free adapter has a structural advantage and where
                          adaptive-MPC / sysID structurally cannot follow.

Every fault is grounded in a documented real F1TENTH failure or sim-to-real gap
(the `mechanism` field) so the injection is defensible as "the real gaps,
deliberately amplified," not invented adversity.

Faults chosen for research interest + cross-paper comparability:
  friction_drop     (in-model, anchor)  -- LLA-MPC 2505.19512 (sudden/gradual),
                                           TC-Driver 2205.09370, Continual-RL 2607.24320
  tire_stiffness    (in-model, anchor)  -- on-track sysID 2411.17508 (hard<->soft)
  steering_loe      (out-of-model)      -- canonical fault-tolerant-control loss-of-effectiveness
  actuator_latency  (out-of-model)      -- dominant documented F1TENTH sim-to-real gap
  steering_bias     (out-of-model)      -- "loose part"; sysID cannot parameterize it

Hooks (all optional; default no-op):
  reset(env)                        -- per-episode state reset
  params(base_params) -> params     -- modify VEHICLE_PARAMS once at reset (in-model)
  step_params(base, t) -> params|None  -- time-varying VEHICLE_PARAMS (e.g. gradual decay)
  command(steer, vel, t) -> (s, v)  -- modify the [steer, vel] actuator command
  observation(obs, t) -> obs        -- modify the observation vector (sensor faults)

Severity is a single scalar in [0, 1] per fault (see `from_spec`) so a monotone
knob can be swept for sensitivity analysis.
"""

import numpy as np


class Fault:
    name = "fault"
    fault_class = "out-of-model"   # or "in-model"
    mechanism = ""

    def reset(self, env):
        return None

    def params(self, p):
        return p

    def step_params(self, base, t):
        return None

    def command(self, steer, vel, t):
        return steer, vel

    def observation(self, obs, t):
        return obs

    def describe(self):
        return f"{self.name} [{self.fault_class}] -- {self.mechanism}"


# --------------------------------------------------------------------------- #
# IN-MODEL faults (perturb VEHICLE_PARAMS; classical adaptation should win)
# --------------------------------------------------------------------------- #

class FrictionDrop(Fault):
    """Global tire-road friction reduction. Modes (matching LLA-MPC 2505.19512
    scenarios): 'sudden' (whole episode at target_mu), 'gradual' (linear decay
    from base_mu to target_mu over ramp_steps), 'sudden_mid' (base_mu until
    onset_frac of episode_steps, then a sudden drop to target_mu)."""
    name = "friction_drop"
    fault_class = "in-model"
    mechanism = "worn tires / dust / wet or polished surface -> lower tire-road mu"

    def __init__(self, target_mu=0.35, base_mu=0.5, mode="sudden", ramp_steps=1500,
                 onset_frac=0.5, episode_steps=8000):
        self.target_mu = float(target_mu)
        self.base_mu = float(base_mu)
        self.mode = mode
        self.ramp_steps = int(ramp_steps)
        self.onset_frac = float(onset_frac)
        self.episode_steps = int(episode_steps)

    def params(self, p):
        if self.mode == "sudden":
            p = dict(p)
            p["mu"] = self.target_mu
        return p

    def step_params(self, base, t):
        if self.mode == "gradual":
            frac = min(1.0, t / max(self.ramp_steps, 1))
            mu = self.base_mu + frac * (self.target_mu - self.base_mu)
        elif self.mode == "sudden_mid":
            onset = self.onset_frac * self.episode_steps
            mu = self.target_mu if t >= onset else self.base_mu
        else:
            return None
        p = dict(base)
        p["mu"] = mu
        return p


class LowGripPatch(Fault):
    """Spatially-varying friction: mu drops to target_mu only while the car is
    within a contiguous span of the reference path (a localized slick patch),
    otherwise base_mu. Mirrors the 'polished-concrete patch' protocol of
    Continual-RL on RoboRacer (2607.24320)."""
    name = "low_grip_patch"
    fault_class = "in-model"
    mechanism = "localized low-grip zone (spilled fluid / wet or polished patch) -> mu drops only within a track region"

    def __init__(self, target_mu=0.2, base_mu=0.5, frac_start=0.35, frac_end=0.55):
        self.target_mu = float(target_mu)
        self.base_mu = float(base_mu)
        self.frac_start = float(frac_start)
        self.frac_end = float(frac_end)
        self._env = None
        self._n = 1

    def reset(self, env):
        if env is None:
            self._env = None
            return
        self._env = env
        self._n = max(len(env.pp.wpts_xy) - 1, 1)

    def step_params(self, base, t):
        if self._env is None:
            return None
        frac = self._env._closest_idx / self._n
        in_patch = self.frac_start <= frac <= self.frac_end
        p = dict(base)
        p["mu"] = self.target_mu if in_patch else self.base_mu
        return p


class TireStiffnessChange(Fault):
    """Scale front/rear cornering stiffness (hard<->soft tire compound)."""
    name = "tire_stiffness"
    fault_class = "in-model"
    mechanism = "different tire compound (hard<->soft) -> changed cornering stiffness C_Sf/C_Sr"

    def __init__(self, scale=0.6):
        self.scale = float(scale)

    def params(self, p):
        p = dict(p)
        p["C_Sf"] = p["C_Sf"] * self.scale
        p["C_Sr"] = p["C_Sr"] * self.scale
        return p


class MassChange(Fault):
    """Added payload / mass change (also shifts effective dynamics)."""
    name = "mass_change"
    fault_class = "in-model"
    mechanism = "added payload -> increased mass m and inertia I"

    def __init__(self, scale=1.3):
        self.scale = float(scale)

    def params(self, p):
        p = dict(p)
        p["m"] = p["m"] * self.scale
        p["I"] = p["I"] * self.scale
        return p


# --------------------------------------------------------------------------- #
# OUT-OF-MODEL faults (command / observation layer; the frontier)
# --------------------------------------------------------------------------- #

class SteeringBias(Fault):
    """Constant additive steering offset -- the 'loose part' archetype."""
    name = "steering_bias"
    fault_class = "out-of-model"
    mechanism = "misaligned/loose steering linkage or servo trim -> constant additive steering offset"

    def __init__(self, bias_rad=0.03):
        self.bias = float(bias_rad)

    def command(self, steer, vel, t):
        return steer + self.bias, vel


class SteeringLossOfEffectiveness(Fault):
    """Commanded steering only partially realized (canonical FTC LoE fault)."""
    name = "steering_loe"
    fault_class = "out-of-model"
    mechanism = "partial steering-actuator failure -> commanded angle only partially realized (FTC loss-of-effectiveness)"

    def __init__(self, loe=0.3):
        self.loe = float(loe)

    def command(self, steer, vel, t):
        return steer * (1.0 - self.loe), vel


class ActuatorLatency(Fault):
    """Apply the command from `delay` steps ago (extra latency beyond the sim's
    own 2-step steer buffer). Affects both steering and velocity commands."""
    name = "actuator_latency"
    fault_class = "out-of-model"
    mechanism = "VESC/servo command latency beyond nominal -> control applied several control-steps late"

    def __init__(self, delay_steps=5):
        self.delay = int(delay_steps)
        self._buf = []

    def reset(self, env):
        self._buf = []

    def command(self, steer, vel, t):
        if self.delay <= 0:
            return steer, vel
        self._buf.append((steer, vel))
        if len(self._buf) <= self.delay:
            return steer, vel          # not enough history yet: pass through
        return self._buf.pop(0)        # command from `delay` steps ago


class WheelDrag(Fault):
    """A dragging / partially-seized wheel: longitudinal drag (commanded velocity
    is not fully realized) plus a constant yaw pull to one side."""
    name = "wheel_drag"
    fault_class = "out-of-model"
    mechanism = "dragging/seizing wheel (bearing or binding brake) -> longitudinal drag + constant yaw pull"

    def __init__(self, vel_loss=0.25, yaw_bias=0.02):
        self.vel_loss = float(vel_loss)
        self.yaw_bias = float(yaw_bias)

    def command(self, steer, vel, t):
        return steer + self.yaw_bias, vel * (1.0 - self.vel_loss)


class ObservationNoise(Fault):
    """Additive Gaussian noise on the observation vector (sensor noise)."""
    name = "obs_noise"
    fault_class = "out-of-model"
    mechanism = "lidar/odometry sensor noise -> corrupted observation vector"

    def __init__(self, std=0.05, seed=0):
        self.std = float(std)
        self._rng = np.random.default_rng(seed)

    def reset(self, env):
        pass

    def observation(self, obs, t):
        if self.std <= 0:
            return obs
        return obs + self._rng.normal(0.0, self.std, size=obs.shape).astype(obs.dtype)


class ObservationLatency(Fault):
    """Return the observation from `delay` steps ago (perception latency)."""
    name = "obs_latency"
    fault_class = "out-of-model"
    mechanism = "lidar/perception pipeline latency -> policy acts on a stale observation"

    def __init__(self, delay_steps=3):
        self.delay = int(delay_steps)
        self._buf = []

    def reset(self, env):
        self._buf = []

    def observation(self, obs, t):
        if self.delay <= 0:
            return obs
        self._buf.append(np.array(obs, copy=True))
        if len(self._buf) <= self.delay:
            return obs
        return self._buf.pop(0)


# --------------------------------------------------------------------------- #
# Container + factory
# --------------------------------------------------------------------------- #

class FaultSet:
    """Composes multiple faults. The env calls these hooks at the right points;
    each hook is a no-op if no member fault implements it."""

    def __init__(self, faults=None):
        if faults is None:
            faults = []
        elif isinstance(faults, Fault):
            faults = [faults]
        self.faults = list(faults)

    def __len__(self):
        return len(self.faults)

    def reset(self, env):
        for f in self.faults:
            f.reset(env)

    def params(self, base):
        p = dict(base)
        for f in self.faults:
            p = f.params(p)
        return p

    def step_params(self, base, t):
        p = dict(base)
        changed = False
        for f in self.faults:
            r = f.step_params(p, t)
            if r is not None:
                p = r
                changed = True
        return p if changed else None

    def command(self, steer, vel, t):
        for f in self.faults:
            steer, vel = f.command(steer, vel, t)
        return steer, vel

    def observation(self, obs, t):
        for f in self.faults:
            obs = f.observation(obs, t)
        return obs

    def describe(self):
        return "; ".join(f.describe() for f in self.faults) if self.faults else "no faults"


# Severity in [0, 1] -> a fault instance with monotone magnitude.
# 0.0 is (near) no-op; 1.0 is severe. Used for sensitivity sweeps.
def from_spec(name, severity, base_mu=0.5):
    s = float(np.clip(severity, 0.0, 1.0))
    if name == "friction_drop":
        return FrictionDrop(target_mu=base_mu - s * 0.35, base_mu=base_mu, mode="sudden")
    if name == "friction_drop_gradual":
        return FrictionDrop(target_mu=base_mu - s * 0.35, base_mu=base_mu, mode="gradual")
    if name == "friction_drop_mid":
        return FrictionDrop(target_mu=base_mu - s * 0.35, base_mu=base_mu, mode="sudden_mid")
    if name == "low_grip_patch":
        return LowGripPatch(target_mu=base_mu - s * 0.40, base_mu=base_mu)
    if name == "tire_stiffness":
        return TireStiffnessChange(scale=1.0 - s * 0.6)
    if name == "mass_change":
        return MassChange(scale=1.0 + s * 0.6)
    if name == "steering_bias":
        return SteeringBias(bias_rad=s * 0.22)          # up to ~12.6 deg (spans clean->failure)
    if name == "steering_loe":
        return SteeringLossOfEffectiveness(loe=s * 0.7)  # up to 70% loss
    if name == "actuator_latency":
        return ActuatorLatency(delay_steps=int(round(s * 12)))
    if name == "wheel_drag":
        return WheelDrag(vel_loss=s * 0.45, yaw_bias=s * 0.05)
    if name == "obs_noise":
        return ObservationNoise(std=s * 0.15)
    if name == "obs_latency":
        return ObservationLatency(delay_steps=int(round(s * 8)))
    raise ValueError(f"unknown fault name: {name}")


IN_MODEL = ["friction_drop", "low_grip_patch", "tire_stiffness", "mass_change"]
OUT_OF_MODEL = ["steering_bias", "steering_loe", "actuator_latency", "wheel_drag",
                "obs_noise", "obs_latency"]
ALL_FAULTS = IN_MODEL + OUT_OF_MODEL


if __name__ == "__main__":
    # Self-test: every fault constructs and its hooks run without error.
    print("Fault-injection self-test")
    for name in ALL_FAULTS:
        f = from_spec(name, 0.5)
        fs = FaultSet(f)
        fs.reset(None)
        p = fs.params({"mu": 0.5, "C_Sf": 4.718, "C_Sr": 5.4562, "m": 3.56, "I": 0.0627})
        sp = fs.step_params({"mu": 0.5, "C_Sf": 4.718, "C_Sr": 5.4562, "m": 3.56, "I": 0.0627}, 100)
        s, v = fs.command(0.1, 3.0, 5)
        o = fs.observation(np.zeros(125, dtype=np.float32), 5)
        print(f"  OK  {f.describe()}")
        print(f"        params.mu={p['mu']:.3f} C_Sf={p['C_Sf']:.3f} m={p['m']:.3f} "
              f"cmd=({s:.3f},{v:.3f}) step_params={'yes' if sp else 'no'}")
    print("all faults constructed and hooks ran.")
