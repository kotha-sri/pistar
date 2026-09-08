# Project Charter — Resilient Fault Adaptation (supersedes `plan.md`)

> **Status (2026-09-08):** Active. Full pivot from lap-time racing RL. **Phase 0 (fault-injection
> feasibility & face validity) is COMPLETE** — see results below. `plan.md` is retained for history.

## Context — why this change

The project began as an RLPP replication: residual RL on Pure Pursuit for F1TENTH, aimed at
self-supervised, few-lap track learning (`plan.md`). That work succeeded on its own terms — the
raceline-reference fix cut deviation 0.65 → 0.17 m and took held-out completion from 6/12 to 12/12,
and Reptile adapts to unseen tracks in ~4K steps (`raceline_campaign_log.md`).

But a deep literature review (Sep 2026) shows **RL as a lap-time optimizer is a dead end**, and —
decisively — our own ablation shows we were **testing RL in the one regime it cannot win**:

- Classical optimal control owns the racing frontier: Tube-MPC won the A2RL full-scale league
  within ~10% of an F1 driver; LMPC laps within ms of the offline optimum; ForzaETH (classical) is
  SOTA on scaled F1TENTH.
- The one rigorous head-to-head (Song et al., *Science Robotics* 2023): *"the fundamental advantage
  of RL over OC is not that it optimizes its objective better, but that it optimizes a better
  objective."* RL only wins under **unmodeled dynamics**.
- The residual-racing lane is crowded (RLPP, ARPO, Drive-Fast-Learn-Faster, all 2025–26); GT Sophy
  (*Nature* 2022) is sim-only with no physical-car transfer.
- **Our tell:** the geometry-vs-dynamics ablation found grip variation *inert*
  (`ablation_geom_vs_dyn.py`) — the diagnosis, not a null result: with an accurate model, linear
  tire regime, and a known raceline, there is nothing for RL to exploit. That is MPC's home turf.

**Decision (full pivot):** retire the lap-time north star. Redirect the same machinery (residual RL
+ Reptile meta-learning + the F1TENTH sim) at the problem where RL is provably the only tool that
works: **online, self-supervised, safe adaptation to out-of-model faults and changes that no model
anticipated** — a loosening part, an asymmetric steering fault, a dragging brake, a degraded
actuator, a novel low-grip patch. Racing/F1TENTH is demoted from "the goal" to "a cheap, safe,
fault-injectable testbed" for a general, embodiment-agnostic capability. Intended outcome: a
defensible contribution (resilient autonomy) plus a publishable contrast (classical wins in-model;
RL wins out-of-model).

Testbed scope: **F1TENTH sim first**; DonkeySim / second-embodiment and real-car transfer are
staged later, not committed here.

---

# Part I — The Case for the Pivot (advisor-facing)

## The reframed thesis

> RL earns its place not as a faster lap-time optimizer, but as the controller that **keeps a robot
> working when it stops matching any model you built** — adapting online, from its own experience,
> to faults and degradations nobody parameterized, safely, in a handful of trials.

A special case of what is widely regarded as **the** reliability bottleneck keeping robots in the
lab. The general problem — resilient / fault-adaptive autonomy — has a distinguished lineage (Cully
*Nature* 2015; Bongard *Science* 2006; RMA, RSS 2021) but is **overwhelmingly studied on legged and
aerial robots**. Wheeled / car-like robots are relatively underserved, and the only vehicle-specific
online-adaptation work is *in-model* surface change (Continual-MAML, 2409.14950). That embodiment
gap plus the specific target below is our lane.

## Why it is novel AND practical

- **Practical:** reliability under the unexpected is what separates a demo from a deployable system;
  the method is embodiment-agnostic and provable on cheap hardware.
- **Novel (corners the well-resourced labs leave open):** (1) adaptation to **out-of-distribution /
  never-seen fault types**; (2) **safety *during* adaptation** on an already-degraded system;
  (3) the **self-supervised degradation signal**; (4) an **underserved wheeled embodiment**.

## The precision claim (state this first, before the advisor does)

"Out-of-model, therefore model-free" is **too strong** — Nagabandi's *learned* model recovers a
crippled robot. The honest claim: the **analytical/Pacejka model is insufficient**; the fix is
online re-fitting, via a learned model *or* a model-free residual. The real competitor is
**learned-model meta-RL (Continual-MAML style)**, not analytical adaptive MPC — and we benchmark
against it.

## What the advisor will push on

- *Crowded field?* → a specific unclaimed corner (OOD faults + safety + wheeled), not out-RMA-ing
  quadruped labs.
- *Why not adaptive MPC / online sysID?* → those re-fit parameters the model already has; helpless
  against faults outside the parameterization — our target.
- *Just domain randomization?* → DR yields one fixed robust policy; we *adapt* — infer the specific
  fault online — covering faults outside the randomized set.

---

# Part II — Technical Approach

## Architecture

```
        +-----------------------------------------------+
 obs -> |  Nominal base: ESTABLISHED F1TENTH controller | -> base action
        |  (tuned Pure Pursuit / MAP / Follow-the-Gap)  |
        +-----------------------------------------------+
        |  Adaptation module (always on): infers a      | -> latent fault context z
        |  latent fault vector from recent history      |
        |  (RMA-style); modulates the residual          |
        +-----------------------------------------------+
        |  Safety guardrail (decoupled): state-pred     | -> clamp / revert to base
        |  residual monitor + fallback-safe fallback    |
        +-----------------------------------------------+
             self-supervised online updates from progress/deviation signal
```

**Design commitments corrected by the prior art:**

1. **Adaptation runs continuously**, not gated by the detector (RMA / quadrotor transformer-FTC have
   no explicit trigger). The detector is an **independent safety guardrail** — decouple "adapt" from
   "stay safe."
2. **Train the corrector in sim with injected faults, never from static recordings** — a recording
   has no counterfactual. Recordings are for the nominal reference, detector calibration, evaluation.
3. **The analytical model is insufficient** → adapt by online re-fitting; benchmark vs learned-model meta-RL.
4. **Adapt on top of established, community-accepted controllers.** Nominal base + comparisons are
   standard F1TENTH practice — tuned Pure Pursuit, MAP (ForzaETH default, 2209.04346), Follow-the-Gap;
   ForzaETH stack (2403.11784) as the "good practice" reference. Results show adaptation rescues *what
   the community already trusts* under faults. (The old Reptile v6 policy is off the critical path.)

## Fault taxonomy (the experimental backbone)

The contribution *is* the contrast between these two classes:

| Class | Definition | Examples | Expected winner |
|---|---|---|---|
| **In-model** | Expressible by perturbing `VEHICLE_PARAMS` | grip μ, mass m, tire stiffness C_Sf/C_Sr | classical adaptive/sysID |
| **Out-of-model** | Not representable by model parameters | steering trim bias, actuator deadband/delay/LoE, wheel drag, low-grip patch, sensor latency/noise | our meta-adaptation residual |

Injection points (implemented in `fault_injection.py` + `residual_env.py`):
- In-model: `RLPPEnv` reset/step → `Simulator.update_params()` (supports per-step for gradual decay).
- Out-of-model: `RLPPEnv.step()` command layer (bias/deadband/delay/LoE) and `_build_obs` (latency/noise).

### Prioritized faults — research interest + cross-paper comparability

| Priority | Fault | Class | Comparable to (adopt their protocol/metric) |
|---|---|---|---|
| **Anchor 1** | Surface-friction drop (gradual decay + sudden drop) | in-model | LLA-MPC 2505.19512 (its two scenarios); Continual-RL 2607.24320 (30–39% lower μ); TC-Driver 2205.09370 (**crash ratio**); Continual-MAML 2409.14950 |
| **Anchor 2** | Tire-model change (hard→soft / stiffness) | in-model | on-track sysID 2411.17508; LLA-MPC (Pacejka) |
| **Frontier 1** | Steering loss-of-effectiveness (30%/50%) | out-of-model | FTC literature (2505.08223 + FTC survey) — recovery/stability |
| **Frontier 2** | Actuator/command latency & lag | out-of-model | documented dominant F1TENTH sim-to-real gap (2506.15899) |
| **Frontier 3** | Asymmetric steering trim/bias ("loose part") | out-of-model | archetypal; clean demo sysID/adaptive-MPC cannot parameterize |

Anchors give directly comparable numbers *and* are the honest control where classical wins; frontier
faults map onto the FTC convention and are where our structural advantage lives.

---

# Part III — Execution Plan (F1TENTH sim first)

Reuse is the default; new files are forks of existing ones.

### Phase 0 — Fault-injection feasibility & realism ✅ DONE (2026-09-08)
Prove the sim can inject realistic faults through existing hooks, before building adaptation on top.

- **Delivered:** `fault_injection.py` (8 composable, severity-parameterized fault channels across
  both classes, each citing the real mechanism it emulates); `residual_env.py` extended with a
  backward-compatible `faults=` kwarg (byte-identical when `None`); `demo_fault_injection.py`
  face-validity demo → `fault_demo_results.json`, `fault_demo_traces.png`.
- **Results (Spielberg, standard Pure Pursuit, velocity_gain=0.5):** clean baseline drives the track
  (92.6% progress, mean|d|=0.027 m, no crash). Severity sweep (progress % of one lap):

  | fault | class | s=0 | 0.25 | 0.5 | 0.75 | 1.0 |
  |---|---|---|---|---|---|---|
  | friction_drop | in | 92.6 | 92.3 | 81.7 | 59.7 | 0.6 |
  | tire_stiffness | in | 92.6 | 92.3 | 92.1 | 81.3 | 0.7 |
  | steering_loe | out | 92.6 | 92.3 | 92.0 | 59.8 | 0.6 |
  | actuator_latency | out | 92.6 | 92.5 | 92.3 | 92.2 | 41.6 |
  | steering_bias | out | 92.6 | 92.5 | 92.5 | 92.3 | 92.2 (mean&#124;d&#124; 0.027→0.118 m) |

- **Exit gate MET:** injection works through existing hooks; ≥3 faults across both classes degrade a
  standard controller with a monotone severity knob (steering_bias degrades *tracking accuracy*
  rather than survival — a useful reminder that faults surface in different metrics). Reproduce:
  `python demo_fault_injection.py --track Spielberg`.
- **Follow-ups:** write the one-page realism justification (per-fault citation of the real mechanism
  — most of the text is already in `fault_injection.py` docstrings); tune severity ranges so each
  frontier fault spans clean→failure; add spatially-varying low-grip patch + wheel-drag channels.

### Phase A — Fault taxonomy build-out
Flesh out remaining channels (wheel drag, low-grip patch, gradual-decay scenarios matching LLA-MPC)
to graded severities; document each. **Exit gate:** full taxonomy injectable and characterized.

### Phase B — The regime proof (the paper's spine)
**New** `eval_fault_matrix.py`: run the baseline matrix across the taxonomy at graded severities.
**Goal:** the crossover — standard/robust baselines hold on in-model faults, degrade on out-of-model;
adaptation recovers the out-of-model cases. **Exit gate:** a matrix showing the crossover.

### Phase C — Meta-adaptation over faults
**New** `train_reptile_faults.py` (fork of `train_reptile.py`): swap the task distribution from
*tracks* to *faults*; reuse snapshot/interpolate/warm-start/eval. **New** `fault_distribution.py`
(analog of `multi_track_env.py`): sample a fault + severity per episode; hold out never-seen fault
types for OOD eval. **Optional** `adaptation_module.py`: RMA-style latent encoder from recent
history. **Exit gate:** recovers held-out in-distribution faults in ≤ a few laps; honest OOD number.

### Phase D — Detector + safety guardrail
**New** `fault_detector.py`: repurpose the `dynamics_model.py` ensemble as a state-prediction residual
monitor (large error = "model is wrong") + fallback-safe clamp/revert-to-base (Sinha & Pavone 2023).
Runs independently of adaptation. **Exit gate:** separates fault onset from hard cornering (ROC);
guardrail prevents crashes the unguarded adapter would cause.

### Phase E — Online self-supervised recovery
Combine C+D: recover from a mid-run fault online from the progress/deviation signal, guardrail
keeping it alive. **Exit gate:** few-lap online recovery on real faults with **zero training crashes**.

### Later (not committed here)
DonkeySim transfer (reuse the Phase-0 obs contract from `plan.md`); second wheeled embodiment for the
generality claim; real-car fault recovery.

## Reuse map

| Need | Reuse | Change |
|---|---|---|
| Residual env / fault layer | `residual_env.py` (`RLPPEnv`, `step`, `_randomize_friction`, `VEHICLE_PARAMS`) | `faults=` added ✅ |
| Fault channels | — | **new** `fault_injection.py` ✅ |
| Nominal base + comparisons | `pure_pursuit.py` (have); MAP + Follow-the-Gap | port MAP/FTG |
| Meta-adaptation engine | `train_reptile.py` (Reptile loop, snapshot/interpolate, eval) | fork → faults |
| Task-distribution wrapper | `multi_track_env.py` | fork → `fault_distribution.py` |
| Detector + learned-model baseline | `dynamics_model.py` (ensemble) | wrap as monitor |
| RL algorithm | sbx `CrossQ` (as in `train_crossq.py`) | none |

## Baseline matrix (rows × the fault taxonomy)

1. **Standard F1TENTH controllers, no adaptation** (tuned PP, MAP, Follow-the-Gap) — the credible floor.
2. Fixed robust RL — domain-randomized over faults, no online adaptation (isolates DR vs adaptation).
3. Learned-model meta-RL (Continual-MAML analog via `dynamics_model.py`) — the *real* competitor.
4. Adaptive-MPC / online-sysID — cite as the in-model champion (LLA-MPC, on-track sysID); implement if time permits.
5. **Ours:** meta-adaptation residual + safety guardrail, on a *standard* controller base.

## Metrics (chosen to line up with existing papers)

- **Crash ratio** (TC-Driver 2205.09370). **Completion + lap time + mean deviation** (RLPP, ForzaETH).
- **Time-to-recover** (LLA-MPC 2505.19512, Continual-RL 2607.24320, on-track sysID 2411.17508 "under a minute").
- **Performance retention.** **Safety** (crashes during adaptation).
- **OOD generalization** (all of the above on held-out, never-trained fault types) — the headline / differentiator.

## Verification (end-to-end)

- Local: `python fault_injection.py` (self-test ✅); `python demo_fault_injection.py` (Phase 0 ✅).
- `eval_fault_matrix.py` on Spielberg reproduces the in-model-vs-out-of-model crossover.
- Remote GPU (`pistar@10.28.177.234`, env `rlpp`, `PYTHONNOUSERSITE=1`, via `run_train.sh`):
  `train_reptile_faults.py` over the fault distribution; held-out fault eval meets the exit gates.

## Risks & open questions

- Crowded, well-resourced field → mitigate with the specific corner, not scale.
- Safety-during-adaptation is genuinely hard; a safety filter limits exploration on a degraded car
  (Evans et al. 2209.11082).
- Right baselines matter → beat learned-model meta-RL, cite adaptive MPC honestly.
- Stretch bet: RMA fast in-distribution adaptation + Cully-style search for out-of-distribution faults.
- **Open (Phase A/B will answer):** does the single-track sim expose enough out-of-model richness, or
  is a higher-fidelity actuator/dynamics layer needed? Phase-0 signal is encouraging — steering LoE,
  latency, and bias all produce distinct, physically-plausible degradations.

## Reading list / citations

**Tier 1:** Song et al., *Optimal Control vs RL*, Science Robotics 2023 (2310.10943); Kumar et al.,
*RMA*, RSS 2021 (2107.04034); Cully et al., *Robots that can adapt like animals*, Nature 2015
(1407.3501); RLPP (2501.17311).
**Tier 2:** Nagabandi et al., *Learning to Adapt*, ICLR 2019 (1803.11347); Finn et al., *MAML*,
ICML 2017 (1703.03400); Bongard et al., *Resilient Machines*, Science 2006; Continual-MAML
(2409.14950).
**Tier 3:** TC-Driver (2205.09370); Drive Fast, Learn Faster (2505.07321); ARPO (2603.12960);
LLA-MPC (2505.19512); On-track SysID (2411.17508); F1TENTH Supervisor (2209.11082); Betz et al.
survey (IEEE OJ-ITS 2022); RoboRacer/F1TENTH survey (2506.15899).
**Tier 4:** Fallback-Safe MPC, CoRL 2023 (2309.08603); Quadrotor FTC + Transformer (2505.08223);
Foundation Models in Robotics review (2604.15395); TTA survey, IJCV 2025; GT Sophy, Nature 2022.

> Verify author lists / exact venues on arXiv before citing the recent ones (ARPO, on-track sysID, Continual-MAML).
