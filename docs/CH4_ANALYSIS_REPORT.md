# Chapter 4 Phase 4 — Tier 1 analysis

Branch: `feature/adaptation` · Repository: `continual-solar-irradiance` · Date: 2026-09-23
Scope: `docs/ch4_conditional_adaptation_plan.md` §7, Tier 1 only.
Campaign: 36 runs, 30.6 GPU-hours, one source checkpoint (`cdbb71ff…`), four RTX 4090 servers.

**Verdict: the policy as the chapter specifies it fails H4.1 and H4.4; the same policy with its
full-model action disabled passes H4.1 at every site.** With the high-error branch updating the
whole network, the conditional policy is worse than sliding-window adaptation at all three sites
and 32 % worse than doing nothing at Folsom, and a year of adaptation triples its error on the
source distribution. The head-only ablation (§7) — identical except that every triggered update
stays in the head — is equivalent to sliding-window adaptation on MAE within ±5 % at all three
sites, and has the lowest MAE of any method at Folsom and Karpicko. Sliding-window adaptation keeps
a small but significant lead on RMSE and forecast skill everywhere.

The published results came from a code path whose high-error action silently trained only the
head, so the method that was actually validated is the head-only one. The three-action policy of
`eq:ch4-update-trigger` was never exercised before this campaign, and it does not survive a year.

## 1. Main comparison (plan §7.2, `tables/ch4_main.csv`, `.tex`)

Pooled over the lit population, MAE and RMSE in W/m², WAPE and SMAPE in %, forecast skill against
Smart Persistence. Populations: Folsom 362 days / 245,538 samples, Karpicko 298 / 179,858,
Poznań 213 / 123,523.

| Site | Method | MAE | RMSE | WAPE | SMAPE | FS |
|---|---|---|---|---|---|---|
| Folsom | Persistence | 35.95 | 86.10 | 12.17 | 15.77 | 0.00 |
| | Static model | 31.51 | 72.60 | 11.19 | 13.54 | 15.68 |
| | MEMO | 33.26 | 72.74 | 11.40 | 13.85 | 15.51 |
| | NOTE | 104.79 | 138.62 | 43.94 | 36.60 | −61.01 |
| | RLS | **31.31** | 71.94 | **11.11** | **13.45** | 16.44 |
| | Window | 31.50 | **69.48** | 11.08 | 13.54 | **19.30** |
| | Few-shot 1d | 31.78 | 73.15 | 11.23 | 14.06 | 15.04 |
| | Few-shot 7d | 37.14 | 74.75 | 12.60 | 15.30 | 13.18 |
| | **Conditional** | **41.68** | 77.21 | 13.85 | 17.48 | 10.32 |
| Karpicko | Persistence | 67.19 | 142.46 | 35.90 | 45.31 | 0.00 |
| | Static model | 80.40 | 140.14 | 48.28 | 51.24 | 1.63 |
| | MEMO | 79.58 | 140.90 | 46.49 | 50.65 | 1.09 |
| | NOTE | 134.07 | 169.51 | 136.72 | 77.33 | −18.99 |
| | RLS | 75.84 | 134.54 | 44.72 | 48.98 | 5.56 |
| | Window | **66.76** | **124.77** | **35.10** | **37.04** | **12.42** |
| | Few-shot 1d | 68.65 | 136.60 | 37.03 | 38.51 | 4.11 |
| | Few-shot 7d | 68.53 | 138.38 | 35.62 | 38.04 | 2.87 |
| | **Conditional** | 68.54 | 128.06 | 36.67 | 39.90 | 10.11 |
| Poznań | Persistence | 67.73 | 138.99 | 29.55 | 39.98 | 0.00 |
| | Static model | 69.56 | 124.08 | 35.06 | 37.67 | 10.73 |
| | MEMO | 68.00 | 122.48 | 33.94 | 36.33 | 11.88 |
| | NOTE | 96.66 | 135.68 | 92.16 | 55.80 | 2.38 |
| | RLS | 66.56 | 120.84 | 33.12 | 36.26 | 13.06 |
| | Window | **58.92** | **111.79** | **27.92** | **29.60** | **19.57** |
| | Few-shot 1d | 69.20 | 124.45 | 37.77 | 35.67 | 10.46 |
| | Few-shot 7d | 63.49 | 121.50 | 32.11 | 31.79 | 12.58 |
| | **Conditional** | 62.68 | 115.37 | 29.67 | 32.65 | 16.99 |

Three patterns survive from the published section. Deployment shift is severe: the frozen model
loses to Smart Persistence on MAE at both Polish sites. Sliding-window adaptation is the strongest
comparator everywhere. The evaluated NOTE implementation degrades accuracy catastrophically,
consistent with a prediction-variance objective whose minimiser is a constant output.

What does not survive is the position of the proposed policy.

## 2. H4.1 for the policy as specified — not supported (`tables/ch4_equivalence.csv`)

Relative MAE of the conditional policy against sliding-window adaptation, 90 % moving-block
bootstrap interval (7-day blocks, 2,000 resamples, paired by date), equivalence margin ±5 %.

| Site | Relative MAE | 90 % CI | Inside ±5 % |
|---|---|---|---|
| Folsom | +32.32 % | [+26.31, +40.62] | **no** |
| Karpicko | +2.65 % | [+1.23, +4.31] | yes |
| Poznań | +6.37 % | [+4.72, +8.54] | **no** |

The criterion requires every site's interval inside the margin. Two of three fall outside, and
both intervals exclude zero on the wrong side, so this is not a near miss: the policy is
*worse* than the comparator, significantly, at every site.

The Folsom result is the serious one. There the conditional policy is not merely behind the best
comparator, it is **32 % worse than the frozen model it started from** (41.68 against 31.51). On
the source domain, where the prior is well matched and there is no shift to absorb, adaptation is
actively destructive.

## 3. Pairwise contrasts (`tables/ch4_dm.csv`)

Diebold-Mariano on daily loss differentials, Newey-West variance at lag 6, Holm-adjusted across
all 21 comparisons. Every comparison is significant at Holm-adjusted *p* < 0.05.

The policy beats only NOTE at all three sites. At Karpicko and Poznań it beats the frozen model,
MEMO, RLS and few-shot on MAE, and loses to Window. At Folsom it loses to everything except NOTE.

## 4. H4.4 for the policy as specified — not supported (`tables/ch4_retention.csv`)

The adapted model, re-scored on the held-out Folsom source stream against the frozen reference on
the same 30 days:

| Deployment site | Frozen MAE | Adapted MAE | Change |
|---|---|---|---|
| Karpicko | 46.86 | 140.59 | **+200.0 %** |
| Poznań | 46.86 | 147.85 | **+215.5 %** |

The plan's bar was 10 %. A year of adaptation does not degrade source competence, it removes it:
the adapted model is three times worse than the model it started from, and worse than Smart
Persistence. The chapter currently lists retention as unmeasured; it is now measured, and it is a
finding rather than a caveat.

The smoke saw +36 % after three days on a two-day slice. The full campaign shows where that
trajectory leads.

## 5. Update cost (`tables/ch4_actions.csv`)

| Site | Considered | No update | Head | Full model | Update rate | Updates/day |
|---|---|---|---|---|---|---|
| Folsom | 240,108 | 70,075 | 145,379 (60.6 %) | 24,654 (10.3 %) | 70.8 % | 470 |
| Karpicko | 175,424 | 47,251 | 93,386 (53.2 %) | 34,787 (19.8 %) | 73.1 % | 430 |
| Poznań | 120,328 | 33,347 | 65,251 (54.2 %) | 21,730 (18.1 %) | 72.3 % | 408 |

Two things follow. The full-model action is not rare — between a tenth and a fifth of decisions —
so the chapter's cost discussion, which assumes it is exceptional, does not describe this policy.
And the update rate, 71–73 %, is far higher than the published Karpicko run's 59.7 %, so the
policy sits much closer to always-update than the chapter's framing suggests.

The monthly action mix does **not** show a runaway feedback loop, and an earlier draft of this
report claimed one without checking. At Folsom the full-model share *falls* to 1.7 % in July,
exactly when the policy is at its worst relative to the frozen model (§6.1). The share tracks the
seasonal scale of absolute error — a fixed 100 W/m² threshold is a high bar on clear summer days —
rather than the state of the model. Whatever is degrading the model is not visibly re-triggering
itself through the threshold.

The clamp sentinel (`scope_changed_if_raw`) is zero in all three campaigns, so the clamped trigger
of `eq:ch4-abs-error` and an unclamped one would have selected identical actions throughout.

## 6. How the degradation develops

### 6.1 It is progressive, not immediate

Folsom, monthly mean daily MAE in W/m², against the frozen model on the same days:

| Month | Conditional | Frozen | Ratio |
|---|---|---|---|
| Jan | 49.41 | 46.60 | 1.06 |
| Feb | 46.29 | 43.82 | 1.17 |
| Mar | 52.10 | 47.51 | 1.27 |
| Apr | 54.26 | 48.89 | 1.57 |
| May | 60.69 | 51.33 | 2.00 |
| Jun | 40.29 | 24.93 | 2.94 |
| Jul | 18.54 | 7.70 | 3.72 |
| Aug | 20.14 | 8.90 | 3.59 |
| Sep | 23.97 | 10.02 | **4.38** |
| Oct | 49.13 | 35.13 | 2.33 |
| Nov | 48.88 | 34.21 | 2.02 |
| Dec | 43.57 | 28.69 | 2.19 |

The policy starts at parity — the first ten days are within ±10 % of the frozen model, sometimes
better — and separates monotonically from it through the first nine months. This is the residual
risk `sec:ch4-adaptation-protocol` explicitly retains: "a trajectory limited at each step
nonetheless drifts over a longer horizon than those evaluated." The campaign has now realised it.
A week-scale evaluation, which is what `tab:ch4-solar-metrics` reports, cannot see this at all.

### 6.2 The cause is the full-model action

Established by the ablation in §7: removing that one branch removes the degradation.

## 7. Head-only ablation

The same policy, same seed, same checkpoint and thresholds, run with `--t2 1e9` so that every
triggered update is a head update. Results under `results/ch4/ablation/`, folded into every table
in `tables/` as the arm *Conditional (head only)*.

| Site | Full model MAE | Head only MAE | Window MAE | Head only FS | Window FS |
|---|---|---|---|---|---|
| Folsom | 41.68 | **31.03** | 31.50 | 18.51 | **19.30** |
| Karpicko | 68.54 | **66.58** | 66.76 | 10.16 | **12.42** |
| Poznań | 62.68 | 59.65 | **58.92** | 17.37 | **19.57** |

**H4.1 for the head-only variant** — relative MAE against Window, 90 % CI, margin ±5 %:

| Site | Relative MAE | 90 % CI | Inside ±5 % |
|---|---|---|---|
| Folsom | −1.48 % | [−1.95, −1.03] | yes |
| Karpicko | −0.27 % | [−0.89, +0.43] | yes |
| Poznań | +1.23 % | [+0.73, +1.84] | yes |

All three intervals sit well inside the margin, so **H4.1 is supported for the head-only policy**.
At Folsom it is significantly *better* than Window on MAE. On squared loss the ordering reverses:
the Diebold-Mariano contrasts favour Window at every site (Holm-adjusted *p* = 0.007, 0.0001,
< 0.0001), with forecast-skill leads of 0.8, 2.3 and 2.2 points. The head-only policy matches
Window on typical errors and trails it on the large ones.

**Where the full-model action does its damage.** Full model against head only is significant at
Folsom (+34 % MAE, Holm *p* < 0.0001) but not at Karpicko (*p* = 0.84) or Poznań (*p* = 0.84),
where RMSE is within 0.4 points. At the shifted sites, updating the encoder roughly breaks even
in-site while destroying source competence (§4); at the source site, where there is no shift to
absorb, moving the encoder is pure damage. The ablation does not include a retention pass, so the
head-only policy's forgetting is not yet measured; with the encoder frozen it is confined to the
head.

**Update rate.** The head-only policy updates on 42.7 % of Folsom decisions against 70.8 % for the
full-model variant, and on 69.0 / 65.2 % against 73.1 / 72.3 % at Karpicko and Poznań. The damaged
model makes larger errors, so it triggers more often. This is a consequence of the damage rather
than a runaway loop — the full-model share itself does not rise as the model degrades (§5) — but
it means the full-model variant also costs more, on top of the cost of each full update.

## 7b. Partial-encoder ablation (layer3 + layer4)

The high-error action trains head + layer3 + layer4 (`--encoder-blocks 2`, 22.6M of 24.0M
parameters), which is what the original code meant by its last two blocks. Thresholds, seed and
checkpoint are unchanged. Retention measured as for the full-model runs.

| Site | Full model | **Layer3+4** | Head only | Window | Layer3+4 retention |
|---|---|---|---|---|---|
| Folsom | 41.68 / 10.32 | 37.43 / 15.42 | **31.03** / 18.51 | 31.50 / **19.30** | — |
| Karpicko | 68.54 / 10.11 | **66.00 / 13.46** | 66.58 / 10.16 | 66.76 / 12.42 | **+194 %** |
| Poznań | 62.68 / 16.99 | 59.96 / **19.89** | 59.65 / 17.37 | **58.92** / 19.57 | **+209 %** |

MAE (W/m²) / forecast skill (%). At both shifted sites layer3+4 is the best conditional variant
and at Karpicko the best method on every metric, including RMSE, where head-only trails Window
by 3 W/m². Updating the deep layers buys a real in-site gain on large errors. It passes H4.1 at
Karpicko and Poznań and fails at Folsom (+18.8 %, CI [+15.0, +23.8]), where it is still 19 %
worse than the frozen model. Its drift is slower than the full model's (September ratio to frozen
2.0 against 4.4) but not arrested, and it forgets almost as badly (+194 %, +209 %).

That is the case for fixing the encoder action rather than dropping it: the gain at shifted sites
is worth keeping if the Folsom damage and the forgetting can be removed. Three single-factor fixes
are implemented (`--replay-size/--replay-k`, `--anchor-pull`, `--encoder-lr`); a fix passes only
if it (i) is no worse than head-only at Folsom and (ii) keeps layer3+4's forecast-skill lead over
Window at the shifted sites. Retention was dropped as a criterion, and H4.4 with it, by author
decision on 2026-09-23.

## 7c. Stabilising the encoder action

One stabiliser per run on the layer3+4 action, everything else unchanged. MAE (W/m²) / forecast
skill (%).

| Variant | Folsom | Karpicko | Poznań | (i) | (ii) |
|---|---|---|---|---|---|
| layer3+4 | 37.43 / 15.42 | 66.00 / 13.46 | 59.96 / **19.89** | ✗ | ✓ ✓ |
| + replay (k = 7) | 37.05 / 13.71 | 66.32 / **14.05** | 60.92 / 19.37 | ✗ | ✓ ✗ |
| + anchor (pull 0.005) | 31.44 / 17.82 | 67.76 / 8.24 | 59.45 / 17.80 | ✗ | ✗ ✗ |
| + encoder LR 1e-6 | 31.16 / **19.23** | **65.90** / 12.32 | **58.58** / 18.91 | ≈ | ✗ ✗ |
| head only | **31.03** / 18.51 | 66.58 / 10.16 | 59.65 / 17.37 | — | ✗ ✗ |
| Window | 31.50 / 19.30 | 66.76 / 12.42 | 58.92 / 19.57 | — | — |

**No stabiliser passes both criteria.** Every configuration whose encoder moves enough to keep the
large-error gain at the shifted sites also damages Folsom, and every one that protects Folsom gives
the gain up.

- *Replay* changes the direction of the encoder's steps but not their size, so the Folsom drift is
  untouched; at Folsom the replayed samples are the same domain and have nothing to restore.
- *Anchor* at the pre-registered strength protects Folsom but pulls too hard to adapt: at Karpicko
  it is the worst adaptive configuration (FS 8.24, below head-only).
- *Low encoder LR* is the best compromise. Against head-only it is significantly better on squared
  loss at all three sites (Holm-adjusted *p* = 0.0009, 0.0001, < 0.0001) and better on MAE at both
  shifted sites, at a cost of 0.13 W/m² MAE at Folsom. Against Window it has lower MAE at every
  site (−1.1, −1.3, −0.6 %), ties on squared loss at Folsom and Karpicko (*p* = 1.0) and trails at
  Poznań (*p* = 0.024). It passes H4.1 at all three sites.

**Best conditional policy on this evidence:** high-error action on head + layer3 + layer4 with an
encoder learning rate of 1e-6. It has the lowest MAE of any method at every site and is safe on the
source domain, but it ties rather than beats sliding-window adaptation on large errors.


## 8. What this does and does not establish

**H4.2 remains untested.** The trigger controls were deferred. With the head-only policy updating
on 43–69 % of decisions, the comparison with always-update and with a rate-matched random trigger
is what would show whether the error threshold does work.

**H4.3 is now arguable but not settled.** The head-only policy matches Window on MAE while updating
single samples rather than four-sample mini-batches, which is the cheaper operation. The device
cost comparison in `tab:ch4-solar-edge-benchmark` timed this head-only behaviour, since that was
the only behaviour the code had, so those numbers describe the head-only policy and remain usable
for it — but daily energy is still unmeasured.

**H4.4 is measured for the full-model variant only**, and fails badly. The head-only variant needs
its own retention pass.

**Single seed, single checkpoint.** The six deterministic methods are bit-identical on a repeat,
so between-run variance is zero by construction; few-shot's three seeds spread by 0.02–0.75 W/m²
(`tables/ch4_seed_spread.csv`), negligible against the effects above. What is uncharacterised is
sensitivity to the source checkpoint, not run-to-run noise.

## 9. Reproducing this analysis

```bash
python scripts/analyze_ch4_tier1.py     # reads results/ch4/tier1 and results/ch4/ablation, writes tables/ch4_*.csv
```

Every estimator is imported from `cirr/fl/analysis.py`, so Chapter 4 and Chapter 5 share one
definition of a bootstrap interval and a Diebold-Mariano test. The only deliberate difference is
the equivalence margin: ±2 % in Chapter 5, ±5 % here per H4.1.
