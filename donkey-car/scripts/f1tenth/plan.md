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
- [ ] **Evaluate zero-shot on held-out tracks (Austin, Silverstone, Monza)**
- [ ] **Evaluate few-lap adaptation (0, 4K, 10K steps on held-out tracks)**
- [ ] Track encoder: context vector from first few observations for rapid adaptation
- [ ] Same mechanism handles new tracks AND new vehicles → DonkeySim and hardware become cheap
- [ ] **Exit gate:** unseen track, competent driving within 5 laps of experience (~10K steps)

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
