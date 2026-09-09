"""Phase D -- fault detector + fallback-safe guardrail (resilient-adaptation pivot).

Two decoupled pieces (per the charter's design: adaptation runs continuously;
the detector is an INDEPENDENT safety guardrail, not the adaptation trigger):

1. ResidualMonitor -- an online "the car isn't behaving nominally" detector. It
   watches a scalar innovation signal (how far the vehicle's actual response
   diverges from the analytical/nominal model's prediction), tracks its EWMA +
   running std, and flags a fault when the standardized residual stays above a
   threshold for `persistence` consecutive steps -- so a single hard corner does
   NOT trip it, but a sustained fault does.

   Default innovation = yaw-rate innovation: the kinematic bicycle model predicts
   yaw_rate = vx/L * tan(steer_cmd); a steering fault (bias / loss-of-effectiveness)
   or a grip fault at the limit makes the OBSERVED yaw_rate diverge from that.
   This needs no trained model. `EnsembleResidualMonitor` (stub) is the richer
   option: use the learned dynamics ensemble (dynamics_model.py) as the predictor.

2. FallbackGuardrail -- attenuates the learned residual toward the base action as
   detector risk rises, and hard-reverts to the base controller (with an optional
   velocity derate) above a hard threshold. This is the "keep the car alive while
   it adapts" piece (ARPO-style attenuation + Sinha & Pavone 2023 fallback-safe).

The two compose but are independent: you can run the monitor for logging only, or
the guardrail off a different risk signal.
"""

import numpy as np

WHEELBASE = 0.174 + 0.151   # lf + lr from VEHICLE_PARAMS


def kinematic_yaw_residual(vx, steer_cmd, yaw_rate_obs, wheelbase=WHEELBASE):
    """Innovation of the nominal kinematic bicycle model: |observed yaw rate -
    predicted yaw rate|. Large + sustained => the model is wrong => a fault."""
    pred = vx / wheelbase * np.tan(steer_cmd)
    return abs(yaw_rate_obs - pred)


def smooth_ewma(x, alpha):
    """Causal EWMA of a 1-D sequence (used to smooth the residual so transient
    corner spikes don't trip the detector; sustained faults still elevate it)."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    s = 0.0
    for i, v in enumerate(x):
        s = v if i == 0 else (1 - alpha) * s + alpha * v
        out[i] = s
    return out


def compute_position_baseline(residuals, pos_fracs, n_bins=50, smooth_alpha=0.02,
                              settle=100, min_count=15):
    """Per-track-position baseline of the SMOOTHED residual. The nominal yaw
    innovation is not stationary around a lap -- it is genuinely high on sharp
    corners even when healthy -- so a fault must be judged against the LOCAL
    nominal, not a global one. Bins the clean run by track fraction and returns
    (bin_mean[n_bins], bin_std[n_bins]); sparse bins fall back to global stats."""
    s = smooth_ewma(residuals, smooth_alpha)[settle:]
    pf = np.asarray(pos_fracs, dtype=float)[settle:]
    bins = np.clip((pf * n_bins).astype(int), 0, n_bins - 1)
    gmean, gstd = float(s.mean()), float(max(s.std(), 1e-6))
    bmean = np.full(n_bins, gmean)
    bstd = np.full(n_bins, gstd)
    for b in range(n_bins):
        m = s[bins == b]
        if len(m) >= min_count:
            bmean[b] = m.mean()
            bstd[b] = max(m.std(), 1e-6)
    return bmean, bstd


class ResidualMonitor:
    """Position-conditioned, smoothed residual monitor with persistence gating.

    Calibrate a per-track-position baseline of the smoothed yaw innovation on a
    clean run (compute_position_baseline), inject it via set_baseline(), then feed
    (residual, pos_frac) each step. A fault is flagged when the smoothed residual
    stays > z_thresh sigma above the LOCAL (position-specific) nominal for
    `persistence` consecutive steps -- so a sharp corner (high but expected
    locally) does not trip it, while a sustained fault does. Position-conditioning
    is what lets it separate faults from hard cornering (the Phase D exit gate)."""

    def __init__(self, z_thresh=4.0, persistence=50, n_bins=50, smooth_alpha=0.02,
                 startup=250):
        self.z_thresh = float(z_thresh)
        self.persistence = int(persistence)
        self.n_bins = int(n_bins)
        self.smooth_alpha = float(smooth_alpha)
        self.startup = int(startup)   # grace period: don't judge during launch transient
        self.reset()

    def reset(self):
        self._s = None                 # smoothed residual (EWMA)
        self._bmean = None
        self._bstd = None
        self._frozen = False
        self._n = 0
        self._streak = 0
        self._fault = False
        self._onset_step = None

    def _smooth(self, residual):
        self._s = residual if self._s is None else \
            (1 - self.smooth_alpha) * self._s + self.smooth_alpha * residual
        return self._s

    def set_baseline(self, bin_mean, bin_std):
        self._bmean = np.asarray(bin_mean, dtype=float)
        self._bstd = np.asarray(bin_std, dtype=float)
        self.n_bins = len(self._bmean)
        self._frozen = True

    def update(self, residual, pos_frac):
        """Feed one raw residual + track fraction; returns {fault, z, streak, onset_step}."""
        self._n += 1
        s = self._smooth(residual)
        if not self._frozen:
            return {"fault": False, "z": 0.0, "streak": 0, "onset_step": None}
        b = min(int(pos_frac * self.n_bins), self.n_bins - 1)
        z = (s - self._bmean[b]) / self._bstd[b]
        if self._n < self.startup:          # ignore the launch transient
            return {"fault": False, "z": float(z), "streak": 0, "onset_step": None}
        if z > self.z_thresh:
            self._streak += 1
        else:
            self._streak = max(0, self._streak - 1)
        if not self._fault and self._streak >= self.persistence:
            self._fault = True
            self._onset_step = self._n
        return {"fault": self._fault, "z": float(z), "streak": self._streak,
                "onset_step": self._onset_step}


class EnsembleResidualMonitor(ResidualMonitor):
    """Richer detector backend: use the learned probabilistic dynamics ensemble
    (dynamics_model.py) as the nominal predictor and monitor the prediction
    innovation (optionally normalized by the ensemble's predictive variance).

    Stub: wire up once a nominal dynamics model is trained. The ResidualMonitor
    thresholding logic above is reused as-is; only the residual source changes.
    """

    def __init__(self, dynamics_model=None, **kw):
        super().__init__(**kw)
        self.model = dynamics_model

    def residual_from_transition(self, state, action, next_state):
        if self.model is None:
            raise NotImplementedError(
                "EnsembleResidualMonitor needs a trained dynamics model; "
                "use ResidualMonitor (kinematic innovation) until then.")
        # pred_mean, pred_var = self.model.predict(state, action)
        # return normalized innovation ||next_state - pred_mean|| / sqrt(pred_var)
        raise NotImplementedError


class FallbackGuardrail:
    """Blend the learned residual toward the base action as risk rises.

    risk_z below `soft`  -> residual passes through unchanged.
    between soft and hard -> residual scaled linearly 1 -> min_scale.
    above `hard`          -> residual * min_scale AND velocity derated (revert-to-base).
    """

    def __init__(self, soft=3.0, hard=5.0, min_scale=0.0, vel_derate=0.7):
        self.soft = float(soft)
        self.hard = float(hard)
        self.min_scale = float(min_scale)
        self.vel_derate = float(vel_derate)

    def scale_for(self, risk_z):
        if risk_z <= self.soft:
            return 1.0, False
        if risk_z >= self.hard:
            return self.min_scale, True
        frac = (risk_z - self.soft) / max(self.hard - self.soft, 1e-6)
        return 1.0 + frac * (self.min_scale - 1.0), False

    def apply(self, base_action, residual, risk_z):
        """base_action, residual: 2-vectors [steer, vel]. Returns the guarded
        combined action [steer, vel]."""
        scale, hard = self.scale_for(risk_z)
        out = np.asarray(base_action, dtype=np.float32) + scale * np.asarray(residual)
        if hard:
            out[1] = out[1] * self.vel_derate    # derate velocity on hard fallback
        return out


if __name__ == "__main__":
    # Smoke/demo: run standard Pure Pursuit clean vs under a steering-LoE fault and
    # show the monitor fires on the fault (and stays quiet on the clean run).
    import os
    from residual_env import RLPPEnv
    import fault_injection as fi

    td = os.path.join(os.path.dirname(os.path.abspath(__file__)), "f1tenth_racetracks")
    ek = dict(alpha_rl=0.0, velocity_gain=0.5, mu_noise_std=0.0,
              velocity_curriculum=False, pp_reference="centerline")

    def rollout(fault, monitor=None, cap=6000):
        """Run PP under `fault`; feed (residual, pos_frac) to `monitor` if given.
        Returns (residuals, pos_fracs, onset_step_or_None)."""
        env = RLPPEnv(track_name="Spielberg", tracks_dir=td, max_laps=1,
                      faults=fault, **ek)
        n_wp = max(len(env.pp.wpts_xy) - 1, 1)
        obs, _ = env.reset(seed=0)
        res_list, pos_list = [], []
        onset = None
        for t in range(1, cap + 1):
            steer_cmd, _ = env.pp.get_action(env._last_pos, env._last_heading)
            obs, r, term, trunc, info = env.step(np.zeros(2, dtype=np.float32))
            res = kinematic_yaw_residual(float(obs[2]), float(steer_cmd[0]), float(obs[4]))
            pf = env._closest_idx / n_wp
            res_list.append(res)
            pos_list.append(pf)
            if monitor is not None:
                st = monitor.update(res, pf)
                if st["fault"] and onset is None:
                    onset = st["onset_step"]
            if info["collision"] or term or trunc:
                break
        env.close()
        return res_list, pos_list, onset

    print("Phase D detector smoke (standard Pure Pursuit on Spielberg):")
    # 1. Calibrate a position-conditioned clean baseline.
    clean_res, clean_pos, _ = rollout(None)
    bmean, bstd = compute_position_baseline(clean_res, clean_pos)
    print(f"  clean baseline calibrated over {len(clean_res)} steps, "
          f"{len(bmean)} position bins")

    # 2. Detect faults against the clean baseline (clean should NOT flag).
    cases = [
        (None, "clean (expect: NO flag)"),
        (fi.from_spec("steering_loe", 0.6), "steering_loe@0.6 (expect flag)"),
        (fi.from_spec("steering_bias", 0.8), "steering_bias@0.8 (expect flag)"),
        (fi.from_spec("friction_drop", 0.7), "friction_drop@0.7 (expect flag)"),
        (fi.from_spec("wheel_drag", 0.7), "wheel_drag@0.7   (expect flag)"),
    ]
    for fault, label in cases:
        mon = ResidualMonitor()
        mon.set_baseline(bmean, bstd)
        _, _, onset = rollout(fault, monitor=mon)
        flag = f"FLAG @step {onset}" if onset else "no flag"
        print(f"  {label:<32s} -> {flag}")

    # Guardrail sanity: residual passes through at low risk, attenuates through
    # the soft band, and hard-reverts (+velocity derate) above the hard threshold.
    print("Guardrail (base=[0.10,4.00], residual=[0.20,1.00]):")
    g = FallbackGuardrail()
    for zr in [1.0, 4.0, 6.0]:
        sc, hard = g.scale_for(zr)
        a = g.apply(np.array([0.10, 4.00]), np.array([0.20, 1.00]), zr)
        print(f"  risk_z={zr:>3}: scale={sc:.2f} hard_fallback={hard} -> action={np.round(a,3)}")
