# Chapter 4 Phase 4 — Tier 1 analysis

Branch: `feature/adaptation` · Repository: `continual-solar-irradiance` · Date: 2026-09-23
Scope: `docs/ch4_conditional_adaptation_plan.md` §7, Tier 1 only.
Campaign: 36 runs, 30.6 GPU-hours, one source checkpoint (`cdbb71ff…`), four RTX 4090 servers.

**Verdict: H4.1 is not supported, and H4.4 fails by a wide margin.** The conditional policy, once
its full-model action is implemented as the chapter describes it, is worse than sliding-window
adaptation at all three sites and worse than doing nothing at Folsom. Adapting for a year destroys
the model on the source distribution. Both findings contradict the published section, and both
trace to the same cause: the published results came from a code path whose high-error action
trained only the head.

This report states what the campaign measured. It does not recommend a rewrite of the section
before the ablation in §6 has been run, because that ablation is what separates "the policy is
wrong" from "the full-model action is wrong".

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

## 2. H4.1 — not supported (`tables/ch4_equivalence.csv`)

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

## 4. H4.4 — not supported (`tables/ch4_retention.csv`)

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
And the update rate is far higher than the published Karpicko run's 59.7 %, which is what a
damaged model does: larger errors cross the thresholds more often, which triggers more updates,
which damages the model further. The dead zone does not arrest that loop.

The clamp sentinel (`scope_changed_if_raw`) is zero in all three campaigns, so the clamped trigger
of `eq:ch4-abs-error` and an unclamped one would have selected identical actions throughout.

## 6. What this does and does not establish

**It does not establish that error-triggered adaptation is a bad idea.** Every measurement here is
of one policy with the full-model action live. The single decisive experiment has not been run:
the same policy with the high-error action disabled, which `--t2` makes a one-line change
(`--t2 1e9` sends every triggered update to the head). That is about 4 GPU-hours for three sites,
and it separates two very different conclusions:

- if head-only adaptation recovers the published behaviour, the finding is that **the full-model
  action is harmful** and the section should either drop it or report it as a negative result;
- if head-only adaptation is also poor, the finding is about the **trigger**, and the published
  numbers were a favourable accident of an under-powered update path.

I recommend running that ablation before any part of the section is rewritten.

**H4.2 remains untested.** The trigger controls were deferred, so nothing here separates "this
policy works" from "updating at this rate works". Given that the update rate is 71–73 %, the
policy is close to always-update, and that comparison now matters more than it did.

**H4.3 cannot be settled favourably.** The accuracy-per-cost argument needs the policy to be
cheaper at comparable accuracy. It is neither: it updates on ~72 % of samples, a fifth of those
through the whole network, and is less accurate than a comparator that updates a four-sample
mini-batch head-only.

**Single seed, single checkpoint.** The six deterministic methods are bit-identical on a repeat,
so between-run variance is zero by construction; few-shot's three seeds spread by 0.02–0.75 W/m²
(`tables/ch4_seed_spread.csv`), which is negligible against the effects above. What is
uncharacterised is sensitivity to the source checkpoint, not run-to-run noise.

## 7. Reproducing this analysis

```bash
python scripts/analyze_ch4_tier1.py     # reads results/ch4/tier1 only, writes tables/ch4_*.csv
```

Every estimator is imported from `cirr/fl/analysis.py`, so Chapter 4 and Chapter 5 share one
definition of a bootstrap interval and a Diebold-Mariano test. The only deliberate difference is
the equivalence margin: ±2 % in Chapter 5, ±5 % here per H4.1.
