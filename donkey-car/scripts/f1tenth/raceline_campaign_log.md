# Raceline-Gap Campaign Log (v3 → v8)

**Goal:** Close the gap between the adapted Reptile meta-learning policy and the optimal
F1TENTH raceline — get the car as close to the ideal line as possible **without** dropping
lap-completion rates — and ideally recover lap-time too.

**Key metric:** `|d_rl|` = mean absolute lateral deviation from the optimal raceline (meters).
Lower is tighter to the ideal line. Completion = laps finished out of 10 per track.
Held-out tracks: **Austin, Monza, Silverstone** (never seen during meta-training).
Locked-in eval operating point: `velocity_gain=0.5, alpha_rl=0.70`.

---

## The system, in one paragraph

Pure Pursuit (PP) follows a path and guarantees safety/bounds; a CrossQ RL policy adds a
**residual** correction on top, scaled by `action_scaling=(steer, velocity)` and blended by
`alpha_rl`. **Reptile** meta-learning trains a weight *initialization* over 68 tracks that
fine-tunes to any new track in a few laps (0 / 4K / 10K adaptation steps). The reward
originally penalized deviation from the **centerline** — which is the root of the whole problem.

---

## What changed each run

| Run | What changed vs previous | Reward | action_scaling | Result |
|-----|--------------------------|--------|----------------|--------|
| **v3** | Baseline (centerline reward) | centerline | (0.05, 1.0) | Works, but sits ~0.64m off the raceline |
| **v5** | Doubled action budget | centerline | **(0.10, 2.0)** | Unstable — abandoned |
| **α_rl sweep** | (eval only, on v3) | — | — | Marginal; floor is structural |
| **velocity sweep** | (eval only, on v3) | — | — | PP not over-braking at gain=0.5 |
| **v6** | **Raceline reward** | **raceline** | (0.05, 1.0) | **Breakthrough** — tightest line, all 3 tracks, but slow |
| **v7** | v6 reward + v5 velocity budget | raceline | **(0.10, 2.0)** | Peak accuracy but completion collapsed |
| **v8** | v6 reward + velocity-only bump | raceline | **(0.05, 1.4)** | Fast again, but loses Monza |

---

## Run-by-run detail

### v3 — baseline (centerline reward)
- **Config:** centerline-referenced reward, `action_scaling=(0.05, 1.0)`.
- **Result:** Clears the adaptation exit gate — competent on unseen tracks within ~2 laps,
  beats PP by 8–10%. But the policy tracks the **centerline**, so it sits **~0.64m off the
  optimal raceline** (|d_rl| ≈ 0.59–0.64m across the three tracks).
- **Diagnosis:** The reward has no term rewarding raceline proximity, so the residual has no
  incentive to leave the centerline. This is the gap the whole campaign targets.

### v5 — doubled action budget (FAILED)
- **Change:** `action_scaling` doubled to **(0.10, 2.0)** — more steering + velocity authority,
  hoping the residual could physically reach the raceline and hit higher speeds.
- **Issues encountered:**
  - **OOM kills** — competing stale processes (a 76-day `guided_train.py`, a 2-day MetaWorld
    run) ate ~37GB RAM. Killed them; tightened JAX cache clearing (`jax.clear_caches()` every
    10 iters instead of 50, added `gc.collect()`).
  - **SubprocVecEnv hangs** — with the larger action scale, parallel env workers deadlocked on
    certain tracks. Fixed by switching to `--inner-envs 1` (DummyVecEnv): stable but 3× slower
    (~206s/iter vs ~62s).
  - **Logging** — `nohup` rebuffered stdout despite `python -u`; added explicit
    `sys.stdout.flush()` calls.
- **Result:** **Inconclusive/failed.** Completion oscillated wildly across checkpoints; Monza
  failed on most; final iter-200 checkpoint collapsed to 0/30. Best was iter-25 (20/30),
  never surpassed. |d_rl| gains marginal and unreliable.
- **Lesson:** Doubled action_scaling alone is **not** the fix — the extra authority
  destabilizes few-lap adaptation.

### α_rl sweep (on v3, eval only)
- Swept the residual blend `alpha_rl ∈ {0.55, 0.70, 0.80, 0.90, 1.0}` at 4K & 10K adapt.
- **α_rl=0.70** is the best-completing point — beats the 0.55 default on lap time on all three
  tracks while holding 10/10. Higher α (0.9–1.0) shaves |d_rl| slightly but crashes at 4K.
- **Gains are marginal (1–2%).** Confirmed the ~0.59–0.64m floor is **structural to the
  centerline reward**, not something α_rl can tune away.

### velocity sweep (on v3, eval only)
- Swept `velocity_gain ∈ {0.3, 0.5, 0.7, 0.9}` × α_rl, adapt & eval with matched gain.
- **Answer to "is PP over-braking?" — No, not at the default gain=0.5.**
  - gain=0.3 *induces* over-braking: tops out at 2.9 m/s, +40–55% lap time, no accuracy gain.
  - gain=0.5: ~4.5–4.7 m/s, best-completing (10/10 all tracks).
  - gain=0.7+: residual **crashes immediately** (DNF in 3–44s).
- Policy tops ~4.7 m/s — **well under the vmax=8.0 ceiling**, so the ceiling isn't binding
  either. The residual is tightly coupled to the gain it was meta-trained on.
- **Lock-in: gain=0.5, α_rl=0.70.**

### v6 — raceline reward (THE BREAKTHROUGH)
- **Change:** Reward now references the **optimal raceline** instead of the centerline —
  `alpha_raceline=1.5`, centerline penalty `alpha_dev` cut 1.0→0.3, `use_raceline_progress=True`.
  `action_scaling` kept at v3's safe (0.05, 1.0). Resumed from v3_best.
- **Issue encountered:** Launch crash — my per-iteration `meta_params_latest.pkl` save ran
  before the checkpoint code created the directory (v3/v5 dirs pre-existed; v6's was new).
  Fixed by adding `os.makedirs(save_dir, exist_ok=True)` before the save; relaunched.
- **Result — first real win.** |d_rl| dropped on **all three** held-out tracks with **10/10
  completion maintained**:

  | Track | v3_best (10/10) | v6_best (10/10) | Gap closed |
  |-------|-----------------|-----------------|------------|
  | Austin | 0.6312m (4K) | **0.6068m** (10K) | **−3.9%** |
  | Monza | 0.6378m (4K)\* | **0.5945m** (10K) | **−6.8%** |
  | Silverstone | 0.5887m (10K) | **0.5656m** (10K) | **−3.9%** |

  \* v3 completes Monza only at 4K (10K DNFs); v6 completes Monza at **both** 4K and 10K.
- **Two caveats:**
  1. **Speed regressed** — v6 tracks the raceline *path* tightly but not its *speed profile*
     (Austin 10K: 118s vs v3's 92s). action_scaling=(0.05,1.0) still caps velocity ~4.7 m/s.
     v6 nails *where*, not *how fast*.
  2. **Meta-training unstable** — completion oscillates; the final iter-200 checkpoint fully
     collapses (0/30). All value is in the **iter-25 `meta_params_best.pkl`** (30/30).
     Best-checkpoint-saving is what preserved it. Cutting α_dev to 0.3 destabilizes the init.
- **Verdict: v6 is the winning config** — the only version to improve |d_rl| on all three
  tracks while holding 10/10.

### v7 — raceline reward + doubled action budget (MIXED / FAILED completion)
- **Hypothesis:** v6's raceline reward + v5's velocity budget (0.10, 2.0) → position accuracy
  *and* speed. Raised `alpha_dev` to 0.6 for stability; resumed from v6_best.
- **Result:** **Peak accuracy ever recorded, but completion collapsed.**
  - Best |d_rl| in the whole project: Austin zero-shot **0.5364m**, Silverstone i100 4K
    **0.5574m @ 100s** (a genuine fast+tight point).
  - **But** training-eval best is only **10/30 laps** — the worst of any version (v6/v3 hit
    30/30). Most 4K/10K adaptation runs DNF. No single checkpoint holds all three tracks.
  - Raising α_dev to 0.6 did **not** fix the instability; the larger action scale dominates.
- **Lesson (re-confirms v5):** doubled action_scaling is fundamentally incompatible with robust
  few-lap adaptation, even with the raceline reward and a raceline-aware starting point.

### v8 — raceline reward + velocity-only bump (fast, but loses Monza)
- **Hypothesis from v7:** the instability is a **steering-axis** problem — (0.10, ·) doubles
  steering authority, and *that* is what crashes adaptation. So keep v6's steering scale (0.05)
  and bump **only** velocity: `action_scaling=(0.05, 1.4)`. Resumed from v6_best.
- **Result — hypothesis half-confirmed:**
  - ✅ **Speed recovered** without the adaptation collapse: Austin 87.6s and Silverstone 97s at
    full 10/10 (v6 was 95s / 137s). The velocity-only bump gave back the lap time v6 lost.
  - ❌ **Loses Monza entirely** (DNF at every checkpoint and adapt level), and |d_rl| is
    slightly *worse* than v6 (drives faster → runs wider).

  | | Austin | Monza | Silverstone |
  |---|--------|-------|-------------|
  | **v6** (accuracy) | 94.7s / 0.66m | 126.8s / 0.62m | 136.6s / **0.56m** |
  | **v8** (speed) | **87.6s / 0.65m** | **all DNF** | 97.3s / 0.62m |

---

## Where things stand

Two complementary champions, still no single winner:

- **v6** = tightest to the raceline, all three tracks complete — but slow.
- **v8** = fast on Austin/Silverstone — but sacrifices Monza and a little accuracy.

**The reward question is settled:** the raceline-referenced reward (v6) is the right objective
and it works. The remaining blocker is **meta-training instability** — across *every* version,
checkpoints oscillate and collapse around iter 100, and Monza is the fragile track that drops
out first. This is independent of action scaling and independent of the reward.

## Recurring operational issues (across all runs)
- **Remote network drops** — the GPU host (`pistar@10.28.177.234`) repeatedly reset SSH
  connections (exit-255 / "connection reset by peer"). The `nohup` training/eval jobs kept
  running through every drop; only visibility was lost, and results persisted in the remote logs.
- **Silent stdout buffering** under `nohup` → fixed with explicit `sys.stdout.flush()`.
- **New-checkpoint-dir crash** on first save → fixed with `os.makedirs(..., exist_ok=True)`.
- **Best-checkpoint-saving is essential** — every run's final checkpoint tends to be collapsed;
  the deployable artifact is always an *early* best checkpoint (v6 iter-25), not the last one.

## Recommended next step (not yet done)
Stabilize the meta-training itself so a **single** checkpoint holds all three tracks — e.g.
meta-update clipping, a lower inner-LR, or keeping α_dev higher — rather than any further
change to the reward or action scaling, both of which are now well-explored.

## Artifacts on the remote (`~/rlpp/`)
- Checkpoints: `reptile/reptile_v{3,5,6,7,8}/meta_params_best.pkl`
- Eval scripts: `eval_v6.py`, `eval_v7.py`, `eval_v8.py`, `eval_alpha_sweep.py`, `eval_velocity_sweep.py`
- Training: `train_reptile.py` (flags: `--reward-alpha-raceline`, `--reward-alpha-dev`,
  `--reward-use-raceline-progress`, `--action-scaling`)
