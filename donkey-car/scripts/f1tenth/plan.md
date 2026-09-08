> ⚠️ **SUPERSEDED (2026-09-08).** The project has pivoted from lap-time racing RL to **resilient
> fault adaptation** — see [`fault_adaptation_plan.md`](fault_adaptation_plan.md), the active
> charter. This file is retained for history: the RLPP replication, Reptile meta-learning, and the
> raceline-reference fix are the groundwork the pivot reuses.

# RLPP Replication & Transfer Plan

**Goal:** Wall-clock efficient self-supervised RL for autonomous racing. Learn new tracks with limited, real-time computation.

**Ladder:** F1TENTH (iterate fast) → DonkeySim (validate transfer) → Real car (prove it works)

---

## Phase 0 — Contracts and Measurement

**Purpose:** Lock the interfaces so nothing downstream silently changes the problem.

- [x] `LidarSpec` / observation contract — both sims consume the same config
  - *Decided: lidar-native end-to-end (DonkeySim has native lidar support)*
  - *For now using Frenet/CTE obs per RLPP paper; lidar comes in Phase 4*
- [x] Canonical env interface — same obs, action, reward, termination
- [x] Base controller as swappable component (Pure Pursuit, PID retained as baseline)
- [ ] **Define the metric:** seconds of wall clock until policy sustains ≥X% of reference lap time with <Y crashes per Z laps, on a held-out track. Pin reference (raceline optimizer or tuned MPC).
- [ ] **Profiler breakdown:** sim / IPC / gradient / Python overhead buckets
- [ ] **Exit gate:** same policy runs against both sims by changing one line; baseline wall-clock number established

---

## Phase 1 — F1TENTH Baseline (RLPP Replication)

**Purpose:** Get a working residual policy in the fast environment.

- [x] Vendored F1TENTH sim (single-track Pacejka, RK4, laser collision)
- [x] Pure Pursuit base controller (`d_la=1.2`, `α_v=0.75`)
- [x] RLPP env: obs=[d, Δψ, vx, vy, r, o_traj] ∈ ℝ¹²⁵, action=residual [δ, v]
- [x] Reward: r_adv + r_speed + scaled (r_dev + r_heading) + r_coll (Eqs. 12–17)
- [x] Friction randomization μ ~ N(0.5, 0.15)
- [x] SAC [256,256], lr=3e-4, buffer=1M
- [x] Velocity curriculum (Gaussian around avg velocity of previous episode)
- [x] 2-lap episode truncation via cumulative Frenet progress
- [x] Training runs on GPU (RTX 4060, ~141 FPS)
- [x] Train to full 2M steps (3h 57m wall clock, 552 episodes)

### 2M Training Results — Spielberg (2024-08-04)

Best model saved at 1.6M checkpoint (eval reward 16,690 ± 2.28).

| Config | Laps | Mean Lap | Best Lap | Avg \|d\| |
|---|---|---|---|---|
| PP baseline (α_rl=0) | 0/10 | — | — | — |
| **Best model, α_rl=0.55** | **10/10** | **56.65 ± 0.04s** | **56.61s** | **0.062m** |
| Best model, α_rl=1.0 | 9/10 | 52.81 ± 0.04s | 52.70s | 0.067m |
| Final model (2M), α_rl=0.55 | 5/10 | 56.40 ± 0.07s | 56.30s | 0.065m |
| Final model (2M), α_rl=1.0 | 6/10 | 52.77 ± 0.05s | 52.69s | 0.074m |

Key observations:
- PP alone cannot complete any laps at μ=0.5 — the RL residual is essential
- α_rl=0.55 is 100% reliable; α_rl=1.0 is ~7% faster but occasionally crashes
- Best checkpoint (1.6M) outperforms final (2M) — slight instability in late training
- Lap times (~53–57s on 338m Spielberg) vs paper's ~14s on ~40m lab track — different track scales, qualitative result matches: RLPP reliably completes laps where PP fails

### Remaining
- [ ] Evaluate on held-out tracks (e.g. Austin, Silverstone)
- [ ] **Exit gate:** completes laps on held-out track; wall-clock number to compare against DonkeyCar baseline

---

## Phase 2 — Transfer Smoke Test (Early, Deliberate)

**Purpose:** Find which gap dominates before investing in optimization.

- [ ] Point Phase 1 policy at DonkeySim with matching `lidar_config`
- [ ] Enumerate and rank transfer gaps: dynamics mismatch, obs geometry, latency, reward scale
- [ ] The policy should at least drive (badly)
- [ ] **Exit gate:** transfer gaps ranked with evidence

**Why now:** every week perfecting a method before this test risks building on an invalid assumption.

---

## Phase 3 — Wall-Clock Efficiency

**Purpose:** Optimize wall-clock and sample efficiency toward few-lap learning.

### Ablation Results (300K steps, Spielberg)

| Method | Wall Clock | FPS | Laps | Mean Lap |
|---|---|---|---|---|
| Baseline (SAC, 1 env, UTD=1) | 33.3m | 150 | 7/10 | 56.56s |
| Vec16 (16 envs, UTD=1) | 2.2m | 2310 | 0/10 | — |
| Hz20 (1 env, 20Hz) | 37.7m | 133 | 10/10 | 58.48s |
| UTD4 (1 env, UTD=4) | ~106m | ~47 | 1/10 | 57.44s |
| CrossQ (16 envs, UTD=20, lr=1e-3) | 5.6m | 899 | 1/10 | 56.93s |

### CrossQ 800K (lr=3e-4) — Best Result

| Metric | Value |
|---|---|
| Wall clock | **17.0m** (vs 33.3m baseline, vs 1.5h full SAC) |
| Laps completed | **9/10** (vs 7/10 baseline) |
| Mean lap time | **56.30s** (vs 56.56s baseline — faster) |
| FPS | 784 |

- [x] Vectorized envs (16 parallel)
- [x] CrossQ (JAX-based, no target networks, BatchNorm critic)
- [x] High UTD ratio (UTD=20 per env step)
- [x] Ablation table with eval results
- [x] **CrossQ 800K outperforms baseline SAC in quality AND wall clock**

### MBPO

- [x] `track_generator.py` — procedural closed tracks with occupancy maps
- [x] `dynamics_model.py` — 5-model probabilistic ensemble (PyTorch, GPU)
- [x] `multi_track_env.py` — random track per episode (20 real + 61 generated)
- [x] `train_mbpo.py` — full MBPO + CrossQ + multi-track pipeline
- [x] Smoke-tested end-to-end on remote GPU
- [x] **v1-v3 all failed** — critic explodes from ~1 to 1e9+ within 30K steps
- [x] **Root cause identified:** CrossQ's BatchNorm critic poisoned by synthetic data in shared replay buffer
- [x] **v4 fix: MixedReplayBuffer** — separate real/synthetic storage, controlled 80/20 sampling ratio per batch. BatchNorm running statistics dominated by real data.
- [x] `mixed_replay_buffer.py` — separate buffer with `add_synthetic()`, `sample_real()`, controlled ratio `sample()`
- [x] Deployed to remote, imports verified
- [x] **v4 smoke test** — 10K steps, critic stays 0.93-1.51 through 4 injection cycles (2500 synthetic). v3 exploded to 28.7M at same point.
- [ ] **v4 full run** — multi-track 100K steps across 70+ tracks with held-out eval
- [ ] **Exit gate:** match CrossQ 800K quality (9/10 laps Spielberg) at <100K real steps

---

## Phase 4 — Generalization and Adaptation (The Contribution) ← CURRENT

**Purpose:** "Learn a new track with limited, real-time computation" — the north star.

**Target:** Residual RL agent on PP base policy adapts to a new track in 1-5 laps, with visible improvement each lap.

### Reptile Meta-Learning (launched 2026-08-19)

**Insight:** instead of learning one policy for all tracks (which fails), learn a weight initialization that adapts quickly to any track via fine-tuning.

- [x] `train_reptile.py` — Reptile meta-learning over CrossQ (v2, with track validation)
- [x] `discover_valid_tracks()` — validates raceline/centerline format, filters 10 bad generated tracks
- [x] Training launched: 200 meta-iterations × 20K inner steps × 4 envs, ~80s/iter
- [x] 68 training tracks (17 real + 51 valid generated), Austin/Monza/Silverstone held out
- [x] **Evaluate zero-shot on held-out tracks (Austin, Silverstone, Monza)**
- [x] **Evaluate few-lap adaptation (0, 4K, 10K steps on held-out tracks)**
- [ ] Track encoder: context vector from first few observations for rapid adaptation
- [x] **Exit gate:** unseen track, competent driving within 5 laps (~4K steps) — **met on Austin & Silverstone** (10/10 laps, 8–11% faster than PP). Monza completes at 4K but is unstable across checkpoints/α_rl (see notes).

#### Held-Out Results (2026-08-20, Reptile v3 best checkpoint)

| Track       | PP-only  | 0-step RL | 4K-adapted | Delta vs PP |
|-------------|----------|-----------|------------|-------------|
| Austin      | 106.5s   | 105.0s (10/10) | 95.6s (10/10) | −10.9s |
| Monza       | 112.2s   | 0/10 (fails)   | 100.8s (10/10) | −11.3s |
| Silverstone | 115.3s   | 119.2s (10/10) | 106.3s (10/10) | −8.9s  |

4K-adapted Reptile beats PP by 8–10% on every held-out track. Exit gate met: competent driving within ~2 laps of experience.

### Raceline Optimization (launched 2026-08-20)

**Problem:** current policy tracks the centerline well (|d_cl|≈0.08m) but is ~0.70m off the optimal raceline. PP follows a synthetic centerline-based path; reward penalizes centerline deviation. The RL residual has no incentive to approach the racing line.

**Approach:** add raceline proximity reward `r_raceline = α * exp(-d_rl / 0.3)` and optionally track progress along the real raceline.

- [x] `eval_ceiling.py` — sweep adaptation steps (0–40K) × α_rl (0.55–1.0) with |d_cl| and |d_rl| metrics
- [x] `residual_env.py` — added `reward_overrides` param, `alpha_raceline` reward, `use_raceline_progress` option
- [x] `train_reptile.py` — added `--reward-alpha-raceline`, `--reward-use-raceline-progress`, `--action-scaling` args
- [x] Raceline reward A/B test: proximity reward DNFs (raceline clips walls), progress-based gives modest 1-2cm improvement
- [x] **Root cause identified:** action_scaling=(0.05, 1.0) physically limits the residual — can't reach raceline 0.64m from centerline, capped at 60% of optimal velocity
- [x] `eval_action_scaling.py` — sweep of steering/velocity scaling: vel_s=2.0 gives 12% speed gain (81.8s vs 89.7s), steer=0.10 reduces |d_rl| by 2-3cm
- [x] **Key insight:** meta-init trained with small scaling can't reliably adapt to larger scaling in 10-40K steps — need to retrain
- [x] **Reptile v5 (2026-08-20/21):** action_scaling=(0.10, 2.0), 200 meta-iterations — **INCONCLUSIVE.** Larger action budget gave more expressive residuals but destabilized training: completion oscillated wildly across checkpoints, Monza failed on most (0/10 on final iter-200 ckpt). Best was iter-25 (20/30 held-out laps), never surpassed.
- [x] **Final |d_rl| eval (v3_best vs v5):** Austin 0.6365→0.6059m (−4.8% zero-shot, but v5 unstable at 4K); Monza 0.6220→0.6211m (−0.1%, v5_i200 DNF); Silverstone no reliable gain. **Verdict: doubled action_scaling is not the fix — instability negates the marginal |d_rl| gains.**

#### α_rl Sweep (2026-08-21, v3_best) — `eval_alpha_sweep.py`

Swept residual blend α_rl ∈ {0.55, 0.70, 0.80, 0.90, 1.0} at 4K & 10K adapt. **α_rl=0.70 is the best-completing point** — beats the 0.55 default on lap time on all three tracks while holding 10/10. Higher α (0.9–1.0) improves |d_rl| slightly but crashes at 4K adapt (only survives with 10K). Gains are marginal (1–2%): the ~0.59–0.64m deviation is a **structural limit of centerline-trained reward**, not something α_rl can tune away.

#### Velocity Optimization (2026-08-21, v3_best) — `eval_velocity_sweep.py`

Swept `velocity_gain` (α_v) ∈ {0.3, 0.5, 0.7, 0.9} × α_rl ∈ {0.55, 0.70}, adapt & eval with matched gain, 4K adapt.

| gain | top speed | Austin | Monza | Silverstone |
|------|-----------|--------|-------|-------------|
| 0.3  | 2.9 m/s   | 10/10, 137–144s (slow) | 10/10, 150–156s (slow) | **0/10 DNF** |
| **0.5**  | **4.5–4.7 m/s** | **10/10, 93–96s** | **10/10, 98–101s** | **10/10, 105–106s** |
| 0.7  | —         | 0/10 DNF (crash) | 0/10 DNF | 0/10 DNF |
| 0.9  | —         | 0/10 DNF | 0/10 DNF | 0/10 DNF |

**Answer to the over-braking question: PP is NOT over-braking at the default gain=0.5.** Lowering to 0.3 *induces* over-braking (2.9 m/s, +40–55% lap time, no |d_rl| benefit); raising to 0.7+ makes the residual policy crash immediately (DNF in 3–44s). The policy tops out at 4.5–4.7 m/s — **well below the vmax=8.0 ceiling**, so the velocity-curriculum ceiling is *not* the binding constraint; the residual policy is tightly coupled to the gain=0.5 base it was meta-trained on. Velocity headroom to the optimal (7+ m/s) can only be unlocked by retraining, not by tuning α_v.

#### Combined recommended operating point (v3_best, 4K adapt)

| Track | Best (gain, α_rl) | Lap time | \|d_rl\| | vs default (0.5, 0.55) |
|-------|-------------------|----------|--------|------------------------|
| Austin      | (0.5, 0.70) | 93.27s  | 0.6309m | −2.33s (−2.4%) |
| Monza       | (0.5, 0.70) | 98.22s  | 0.6378m | −2.62s (−2.6%) |
| Silverstone | (0.5, 0.70) | 104.52s | 0.5985m | −1.76s (−1.7%) |

**Lock-in: gain=0.5, α_rl=0.70** — uniform winner, ~2% faster than the current default with 10/10 completion preserved on all three held-out tracks.

- [x] **Reptile v6 (2026-08-23/24):** raceline-referenced reward (α_raceline=1.5, α_dev=0.3, raceline-progress), action_scaling=(0.05, 1.0). **The structural fix worked — first meaningful |d_rl| gain with completion intact.**

#### v6 vs v3 |d_rl| (2026-08-24, best-completing config, gain=0.5, α_rl=0.70) — `eval_v6.py`

| Track | v3_best (10/10) | v6_best (10/10) | Gap closed |
|-------|-----------------|-----------------|------------|
| Austin      | 0.6312m (4K)  | **0.6068m** (10K) | **−3.9%** |
| Monza       | 0.6378m (4K)* | **0.5945m** (10K) | **−6.8%** |
| Silverstone | 0.5887m (10K) | **0.5656m** (10K) | **−3.9%** |

\* v3 completes Monza only at 4K (10K DNFs); v6 completes Monza at **both** 4K and 10K.

**First time |d_rl| dropped on all three held-out tracks with 10/10 completion maintained everywhere.** The raceline reward broke through the ~0.59–0.64m centerline-reward floor that α_rl and velocity sweeps could not.

**Two caveats:**
1. **Speed regressed** — v6 tracks the raceline *path* tightly but not its *speed profile* (Austin 10K 118s vs v3 92s). action_scaling=(0.05,1.0) still caps velocity ~4.7 m/s vs optimal 7+ m/s. v6 nails *where*, not *how fast*.
2. **Meta-training is unstable** — completion oscillates and the final iter-200 checkpoint fully collapses (0/30, like v5). All value is in the **iter-25 `meta_params_best.pkl`** (30/30); best-checkpoint-saving is what preserved it. Cutting α_dev to 0.3 destabilizes the meta-init.

- [x] **Reptile v7 (2026-08-24/25):** v6 raceline reward + v5 velocity budget action_scaling=(0.10, 2.0), α_dev raised to 0.6, resumed from v6_best. **Mixed verdict — record raceline accuracy, but completion collapsed.**

#### v7 vs v6 vs v3 |d_rl| (2026-08-25, gain=0.5, α_rl=0.70) — `eval_v7.py`

Best |d_rl| ever recorded, but only in isolated checkpoint/adapt cells — not robustly:

| Track | Best v7 10/10 cell | \|d_rl\| | Time | vs v6 best |
|-------|--------------------|--------|------|------------|
| Austin      | i50, zero-shot | **0.5364m** | 160s (slow) | −9% \|d_rl\|, but +55s |
| Monza       | i50, zero-shot | 0.6214m | 169s (slow) | worse than v6 |
| Silverstone | i100, 4K       | **0.5574m** | **100s** | −1% \|d_rl\|, fast — **clean win** |

**What v7 proved:** the larger action budget + raceline reward can pull *tighter* to the line than anything before (Austin zero-shot 0.5364m, best in project). Silverstone i100 4K is a genuine fast+tight+10/10 point (0.5574m @ 100s).

**Why it's not the answer:** completion collapsed — v7's training-eval best is only **10/30 total laps** (worst of any version; v6/v3 hit 30/30). The (0.10, 2.0) action budget destabilizes *adaptation* — most 4K/10K runs DNF, and no single checkpoint holds all three tracks. Raising α_dev to 0.6 did **not** fix the instability; the larger action scale dominates. This re-confirms the v5 lesson: **doubled action_scaling is fundamentally incompatible with robust few-lap adaptation, even with raceline reward.**

**Standing conclusion: v6 remains the winning config** — the only version that improved |d_rl| on all three tracks while holding 10/10. v7 is a research data point (peak accuracy is reachable) not a deployable policy.

- [ ] **v8 (next candidate):** keep v6's small action budget (0.05, 1.0) but add a modest *velocity-only* scale bump (e.g. (0.05, 1.4)) to recover lap time without the steering instability that (0.10, ·) causes. OR: stabilize meta-training directly (meta-update clipping / lower inner-LR) so a single checkpoint holds all tracks. Velocity instability is a steering-axis problem — isolate it.

---

## Phase 5 — Hardware

**Purpose:** Only meaningful once Phase 4 works — hardware is just another sample from the training distribution.

- [ ] **Pre-gate:** does the real track have vertical boundaries a 2D lidar can see? (tape on floor = lidar is useless)
- [ ] **Pre-gate:** is the venue indoor? (RPLidar degrades in direct sunlight)
- [ ] If lidar is viable: RPLidar A1/A2, downsample to 36–108 beams, ±90–135° forward arc
- [ ] De-skew scans using wheel odometry/IMU (scan skew at 5+ m/s is the biggest sim-to-real gap)
- [ ] Domain randomization already baked in from Phase 4: per-beam dropouts, range noise, mounting error, latency

---

## Key Principles

1. **Phase 0 defines the invariants. Phases 1–5 cannot change them.** If Phase 3 wants a different observation, that's a Phase 0 change with a re-run of Phase 2.
2. **DonkeySim stays in the loop continuously after Phase 2** — periodic transfer check, not a final exam.
3. **F1TENTH = iteration environment; DonkeySim = validation environment; car = proof.** Never invert.
4. **Measure before optimizing.** Profile wall clock before touching the algorithm.
5. **The fast-adaptation framing** (meta-learning over track distribution) is a stronger contribution than "we made SAC converge 3x faster."
