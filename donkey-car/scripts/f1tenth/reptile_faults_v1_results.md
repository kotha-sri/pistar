# Phase C v1 — Reptile meta-adaptation over faults (results)

**Run:** `train_reptile_faults.py --meta-iterations 200 --inner-steps 8000 --inner-envs 4 --utd 20
--tag reptile_faults_v1` on pistar (RTX 4060). 200 meta-iters, **5 h 51 m**, clean exit, no errors.
8 training fault types, 3 held-out OOD types, severity band 0.2–0.6, eval track Austin, centerline PP.

**Best-score history (held-out recovery, summed completions):** 3 → 10 → 12 → **13**. The meta-loop
is learning.

## The result in one line
Meta-training over a fault distribution produced, *at its peak (iter 175)*, a policy that completes
**3/3 laps zero-shot on all 7 held-out fault tasks — including 3 fault TYPES it never trained on**
(`mass_change`, `obs_latency`, `friction_drop_mid`). That robust zero-shot generalization to unseen
fault types is the headline. But training is **unstable**: the final (iter-200) checkpoint collapsed.

## Eval trajectory (completions at 0 / 2K / 4K adapt steps)

**iter 175 — the peak (strong zero-shot):**

| task | class | 0 | 2K | 4K |
|---|---|---|---|---|
| friction_drop@0.50 | ID | 3 | 2 | 0 |
| low_grip_patch@0.50 | ID | 3 | 0 | 3 |
| tire_stiffness@0.50 | ID | 3 | 3 | 0 |
| steering_loe@0.50 | ID | 3 | 0 | 3 |
| friction_drop_mid@0.40 | OOD | 3 | 0 | 1 |
| obs_latency@0.40 | OOD | 3 | 3 | 3 |
| mass_change@0.40 | OOD | 3 | 0 | 3 |

**iter 200 — the final checkpoint (zero-shot collapsed):**

| task | class | 0 | 2K | 4K |
|---|---|---|---|---|
| friction_drop@0.50 | ID | 0 | 0 | 0 |
| low_grip_patch@0.50 | ID | 0 | 0 | 3 |
| tire_stiffness@0.50 | ID | 0 | 0 | 0 |
| steering_loe@0.50 | ID | 0 | 0 | 0 |
| friction_drop_mid@0.40 | OOD | 0 | 0 | 1 |
| obs_latency@0.40 | OOD | 0 | 3 | 2 |
| mass_change@0.40 | OOD | 0 | 3 | 3 |

## Two problems v1 exposed (both fixed for v2)

1. **Meta-instability / late-training collapse.** Peaks ~iter 150–175, then the iter-200 checkpoint
   degrades to 0/7 zero-shot. Same pattern as the v5/v7 track runs. Few-lap adaptation is also noisy:
   2K/4K fine-tuning frequently *hurts* (e.g. iter-175 `friction_drop` 0-adapt 3/3 → 4K 0/3).
   *Fix (v2):* `--meta-clip` (global-L2-norm clip on the Reptile delta) + lower `--inner-lr` +
   lower `--epsilon-end`.
2. **Wrong checkpoint-selection metric.** v1's `best_score` summed completions at the *4K-adapt*
   budget only, so it saved iter-150 (which relied on 4K adapt) and treated iter-175's superb
   zero-shot as merely tied — the *robust* checkpoint was not the one preserved.
   *Fix (v2):* score now weights **zero-shot (0-adapt) completion 2×** plus the adapted budgets, so
   "best" tracks robustness.

## v2 launch (once pistar is reachable)
```
source ~/miniconda3/etc/profile.d/conda.sh && conda activate rlpp && export PYTHONNOUSERSITE=1
cd ~/rlpp && nohup python train_reptile_faults.py --meta-iterations 200 --inner-steps 8000 \
    --inner-envs 4 --utd 20 --inner-lr 5e-4 --epsilon-end 0.05 --meta-clip 5.0 \
    --tag reptile_faults_v2 > reptile_faults_v2.log 2>&1 &
```
Also worth a direct eval of `meta_params_0175.pkl` (the qualitatively-best v1 checkpoint) to confirm
it as the v1 keeper.
