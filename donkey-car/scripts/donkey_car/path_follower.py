"""
path_follower.py
----------------
Algorithmic track-following baseline for the Donkey Car simulator.

Steering — PID controller on Cross-Track Error (CTE) keeps the car on the
           track centerline.
Throttle — simple proportional controller that chases a target speed.

CTE is how far left/right the car is from the track centerline:

    Positive CTE  →  car is right of center  →  steer left  (negative steering)
    Negative CTE  →  car is left of center   →  steer right (positive steering)

CTE source
----------
In the simulator, CTE is handed to us directly in the telemetry packet
('cte' field) — it is ground truth. On a real car there is no such signal:
you would estimate it from a camera/lane-detector or from localization
against a map. main.py reads telemetry['cte'] and passes it in here, so the
controller itself never touches the sim; swapping in a real-world CTE
estimator means changing only where the value comes from, not this file.

This is meant to be the stable "expert" policy that an RL agent later refines
(residual RL / RL fine-tuning) — so keep it simple and predictable.
"""

from pid import PIDController


# PID + speed tuning. Lives next to the controller so it can be hot-reloaded
# from main.py while the car is driving (see main.py's reload loop).
CONFIG = {
    # Steering PD on cross-track error, retuned for CLEAN, NON-OSCILLATING
    # tracking across the whole speed range the RL residual drives (~0.8–2.0),
    # not just the slow design speed. This is the smooth "slate" the residual
    # sits on top of: a steady offset is trivial for the residual to trim, but
    # base oscillation is toxic to it, so this prioritises minimal wobble/jitter
    # over tight centering. Gentle gains (kp 1.3 / kd 1.2) plus the speed
    # schedule below give, at target_speed 1.6: ~4% centerline weaving and
    # steer-jerk ~0.056 (both ~half the old kp2.5/kd1.6 tuning), max offset
    # ~1.9 m, zero off-track resets at 0.8/1.6/2.0. Chosen from a 3-way parallel
    # gain sweep (low-gain vs schedule-heavy vs high-damping); low-gain won on
    # smoothness. Speed is the RL policy's job.
    "kp" : 1.30,    # proportional: gentle — over-correction is what caused weave
    "ki" : 0.00,    # integral:     LEAVE AT 0 — windup on long curves drove the
                    #               car off track in every trial that used it
    "kd" : 1.20,    # derivative:   light damping; higher kd amplified telemetry
                    #               noise into wheel jitter in every sweep

    # Throttle (proportional speed controller). Kept deliberately slow so the
    # base is stable everywhere; the RL residual adds speed on top via throttle.
    "target_speed" : 0.8,   # desired speed, same units as telemetry 'speed'
    "throttle_kp"  : 1.0,   # how hard to chase target_speed

    # Anti-windup clamp on the integral term. None = no clamp.
    # Only relevant if you reintroduce ki (not recommended — see above).
    "integral_limit" : None,

    # ── Speed-scheduled steering gains ──────────────────────────────────────
    # The PD above was tuned at target_speed 0.8. The RL residual adds throttle
    # and drives the car much faster (~1.7+), where those same gains are
    # underdamped: the P term saturates and the car bang-bangs around the
    # centerline → straightaway weave. Soften the proportional gain as speed
    # rises so the base stays stable across the residual's whole speed range:
    #
    #   kp_eff = kp * (sched_ref_speed / max(speed, sched_ref_speed)) ** sched_kp_exp
    #
    # Speed is floored at sched_ref_speed, so gains are only ever REDUCED above
    # the design speed, never amplified below it — slow-speed behaviour is
    # identical to the validated tuning. Set sched_kp_exp = 0 to disable.
    "sched_ref_speed" : 0.8,   # speed the PD gains were tuned at; schedule pivots here
    "sched_kp_exp"    : 1.0,   # soften kp with speed (0 = off, 1 = ~1/speed)
    "sched_kd_exp"    : 0.0,   # soften kd with speed (0 = keep full damping at speed)
}


class PathFollower:
    """
    Converts CTE + speed telemetry into steering and throttle commands.

    steering : PID on CTE, clamped to [-1.0, 1.0]
    throttle : proportional speed controller, clamped to [0.0, 1.0]
    """

    def __init__(self):
        # Nominal (design-speed) gains. The PID's live kp/kd are overwritten
        # each tick by the speed schedule in compute_controls(); these hold the
        # unscaled values the schedule is computed from.
        self.kp_nom = CONFIG["kp"]
        self.kd_nom = CONFIG["kd"]
        self.pid = PIDController(
            CONFIG["kp"],
            CONFIG["ki"],
            CONFIG["kd"],
            integral_limit=CONFIG["integral_limit"],
        )
        self.target_speed = CONFIG["target_speed"]
        self.throttle_kp  = CONFIG["throttle_kp"]

        # Speed-scheduling parameters (see CONFIG).
        self.sched_ref_speed = CONFIG["sched_ref_speed"]
        self.sched_kp_exp    = CONFIG["sched_kp_exp"]
        self.sched_kd_exp    = CONFIG["sched_kd_exp"]

    def get_file_name(self) -> str:
        """Return this file's name so main.py can hot-reload its CONFIG."""
        return "path_follower.py"

    # ── Control ──────────────────────────────────────────────────────────────

    def compute_controls(self, cte: float, speed: float) -> tuple[float, float]:
        """
        Given the current CTE and speed, return (steering, throttle).

        Parameters
        ----------
        cte   : Cross-track error (positive = car is right of center).
        speed : Current vehicle speed.

        Returns
        -------
        steering : float in [-1.0, 1.0]   (negative = left | positive = right)
        throttle : float in [ 0.0, 1.0]   (0 = stopped, 1 = full forward)
        """
        # Speed-schedule the steering gains: soften kp (and optionally kd) as
        # speed climbs above the design speed so the base stays damped at the
        # higher speeds the RL residual reaches. Floored at sched_ref_speed, so
        # ratio <= 1 always — gains are only reduced, never amplified.
        ratio = self.sched_ref_speed / max(speed, self.sched_ref_speed)
        self.pid.kp = self.kp_nom * (ratio ** self.sched_kp_exp)
        self.pid.kd = self.kd_nom * (ratio ** self.sched_kd_exp)

        # Negate so the car steers back toward center:
        #   too far right (+CTE) → steer left (−steering)
        steering = -self.pid.compute(cte)
        steering = max(-1.0, min(1.0, steering))   # full [-1, 1] range

        # Proportional speed controller — no reverse.
        throttle = self.throttle_kp * (self.target_speed - speed)
        throttle = max(0.0, min(1.0, throttle))

        return steering, throttle

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def reset(self):
        """Clear PID state. Call whenever the car is reset to the start."""
        self.pid.reset()

    def update_config(self, cfg: dict):
        """
        Apply a (partial) CONFIG dict at runtime — used for hot-reloading
        gains while driving. Missing keys keep their current value.
        """
        # Update the NOMINAL gains (the schedule rescales these each tick).
        self.kp_nom = cfg.get("kp", self.kp_nom)
        self.kd_nom = cfg.get("kd", self.kd_nom)
        self.pid.ki = cfg.get("ki", self.pid.ki)
        self.pid.integral_limit = cfg.get("integral_limit", self.pid.integral_limit)
        self.target_speed = cfg.get("target_speed", self.target_speed)
        self.throttle_kp  = cfg.get("throttle_kp",  self.throttle_kp)
        self.sched_ref_speed = cfg.get("sched_ref_speed", self.sched_ref_speed)
        self.sched_kp_exp    = cfg.get("sched_kp_exp",    self.sched_kp_exp)
        self.sched_kd_exp    = cfg.get("sched_kd_exp",    self.sched_kd_exp)
