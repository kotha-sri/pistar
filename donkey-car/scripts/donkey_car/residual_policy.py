"""
residual_policy.py
------------------
Residual Policy Learning (RPL) wrapper, after Trumpp et al., "Residual Policy
Learning for Vehicle Control of Autonomous Racing Cars" (F1TENTH, 2023).

The idea: keep a stable, classical *base* controller doing the hard job of
staying on the track, and let an RL agent learn only a small, bounded
*residual* action that nudges the base toward higher performance (more speed,
smoother turns). The combined command is

    a = clip( a_base  +  scale · clip(a_residual, -1, 1) ,  valid_range )      (paper Eq. 7)

Because the residual is clipped to a small `scale`, it can only *amend* the
base — it can never override it and send the car straight off the track. That
is what makes a thin "go fast / turn smoothly" reward safe to optimize: the
base guarantees on-track behaviour, the residual optimises performance.

This class is deliberately framework-agnostic:
  * the BASE policy is any object exposing
        compute_controls(*base_inputs) -> (steering, throttle)
    (e.g. our PathFollower — the PD-on-CTE controller).
  * the RESIDUAL actor is any callable
        actor(obs: np.ndarray) -> np.ndarray   # shape (2,), values in [-1, 1]
    e.g. a Stable-Baselines3 SAC model's predict(), a raw torch network, or
    the default zero-actor (which makes this behave EXACTLY like the base
    controller until a residual is trained and attached).
"""

from __future__ import annotations

import numpy as np


def _zero_actor(obs: np.ndarray) -> np.ndarray:
    """Default residual: do nothing — combined action == base action."""
    return np.zeros(2, dtype=np.float32)


class ResidualPolicy:
    """
    Wrap an external base policy and add a bounded, learned residual.

    Parameters
    ----------
    base_policy    : object with compute_controls(*base_inputs) -> (steer, throttle).
    steer_bound    : max magnitude the residual may add to steering, in the sim's
                     [-1, 1] steering units. Paper used ±0.05 (of the F1TENTH range).
    throttle_bound : max magnitude the residual may add to throttle, in [0, 1] units.
    residual_actor : callable(obs)->np.ndarray(2,) in [-1, 1]; defaults to zeros.
    allow_braking  : if True, the combined throttle may go negative (reverse/brake);
                     if False (default) it is clamped to [0, 1] like the base.
    """

    def __init__(
        self,
        base_policy,
        steer_bound: float = 0.05,
        throttle_bound: float = 0.15,
        residual_actor=None,
        allow_braking: bool = False,
    ):
        self.base_policy    = base_policy
        self.steer_bound    = float(steer_bound)
        self.throttle_bound = float(throttle_bound)
        self.residual_actor = residual_actor or _zero_actor
        self.allow_braking  = allow_braking

    # ── Components ─────────────────────────────────────────────────────────────

    def base_action(self, *base_inputs) -> np.ndarray:
        """Run the external base policy → np.array([steering, throttle])."""
        steering, throttle = self.base_policy.compute_controls(*base_inputs)
        return np.array([steering, throttle], dtype=np.float32)

    def scale_residual(self, residual) -> np.ndarray:
        """Clip the raw [-1, 1] residual and scale it by the per-axis bounds."""
        r = np.clip(np.asarray(residual, dtype=np.float32).reshape(2), -1.0, 1.0)
        return np.array(
            [r[0] * self.steer_bound, r[1] * self.throttle_bound],
            dtype=np.float32,
        )

    def combine(self, base: np.ndarray, residual) -> np.ndarray:
        """a = clip(base + scaled_residual) into valid sim ranges."""
        a = np.asarray(base, dtype=np.float32).reshape(2) + self.scale_residual(residual)
        a[0] = np.clip(a[0], -1.0, 1.0)                                  # steering
        a[1] = np.clip(a[1], -1.0 if self.allow_braking else 0.0, 1.0)   # throttle
        return a

    # ── Inference (driving) ────────────────────────────────────────────────────

    def act(self, obs: np.ndarray, *base_inputs) -> np.ndarray:
        """
        Combined action for driving with the (trained) residual attached.

        obs         : observation vector the residual actor expects.
        base_inputs : inputs the base policy expects (e.g. cte, speed).
        """
        base     = self.base_action(*base_inputs)
        residual = self.residual_actor(obs)
        return self.combine(base, residual)

    # ── Wiring helpers ─────────────────────────────────────────────────────────

    def set_actor(self, actor) -> None:
        """Attach a residual actor: any callable(obs)->np.ndarray(2,) in [-1, 1]."""
        self.residual_actor = actor

    def attach_sb3(self, model, deterministic: bool = True) -> None:
        """Convenience: use a Stable-Baselines3 model as the residual actor."""
        self.set_actor(lambda obs: model.predict(obs, deterministic=deterministic)[0])

    def reset(self) -> None:
        """Reset the base policy's internal state (e.g. PID integrators)."""
        if hasattr(self.base_policy, "reset"):
            self.base_policy.reset()
