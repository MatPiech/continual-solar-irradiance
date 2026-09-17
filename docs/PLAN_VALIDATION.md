# Phase 0 — Plan validation against the code

Branch: `ch5-benchmarks` · Repository: `continual-solar-irradiance` · Date: 2026-09-17
Scope: read-only discovery for `docs/ch5_server_benchmark_plan.md`. No source file was modified.

---

## 0. Headline findings

Five findings change what Phase 1 has to do. They are stated first because three of them
invalidate parts of the Chapter 5 text as written, not just parts of the plan.

**H-1. The encoder is EfficientViT-B2, not ResNet50.** The checkpoint behind every number in
Table 5.3 is `checkpoints/epoch_024.ckpt`, whose stored hyper-parameters are
`model_name='timm-efficientvit_b2', loss_function='Berhu'`. The chapter's Method paragraph
("A ResNet50 encoder … pooled feature vector … 2052-dimensional representation") does not
describe the runs. The actual pooled width is 2560, the fusion input is 2564, and the head is
2564 → 256 → 1. ResNet50 checkpoints do exist in the repository (`epoch_010.ckpt`,
`resnet50_epoch_015.ckpt`, `epoch_007.ckpt`) but none of them produced the Chapter 5 table.

**H-2. Prototypes are 384-dimensional, not 2048.** For architectures exposing
`image_encoder.stages` (EfficientViT does), prototypes are taken from a forward hook on
`stages[3]` followed by global average pooling, which is 384-d for B2 — not the pooled encoder
output. So the exchanged object is `C × d = 3 × 384 = 1152` values, not 6144. This *resolves*
the chapter's open QUERY about payload bytes: 1152 values × 4 B × 2 directions ≈ 9.2 kB plus
`torch.save` framing ≈ the measured 0.013 MB round trip. The full-model side is
22,458,321 state-dict elements ≈ 89.8 MB one way, 179.7 MB round trip analytically against
171.676 MB measured. Both measured numbers are consistent with fp32 and round-trip accounting;
the chapter's inference of "≈ 2.15 × 10⁷ parameters" was right, its `d = 2048` was not.

**H-3. Night samples are in the evaluation set, unequally across sites.** Counting samples whose
Ineichen–Perez clear-sky GHI is exactly zero: Folsom **0.0 %** (0 / 246,164), Karpicko **18.8 %**
(42,554 / 226,114), Poznań **30.8 %** (56,353 / 183,044). Folsom's camera only records 06–20 h;
the Polish cameras record around the clock. MAE and RMSE in `compute_metrics` are computed over
*all* samples with no threshold (only WAPE/SMAPE apply `min_threshold=0.005`). Table 5.3 therefore
compares a site whose error population is entirely daylight against two sites whose populations
are one fifth and one third trivially-zero. F4 is consequently not a tidy-up: it will move the
Karpicko and Poznań rows and leave Folsom nearly unchanged, and every relative-difference claim
in §5.3.3 has to be recomputed afterwards. (Cross-check: 42,554 is exactly the "naive persistence
fallbacks" count in `results/cl_benchmark/karpicko/karpicko_smart_persistence_15min_results.txt`.)

**H-4. The published runs used two replay buffers that the chapter never mentions.** Every
`summary.json` under `results/fl_benchmark/days-1_warmup30_sync7/` records
`replay_buffer_size: 10000, replay_ratio: 0.5, recent_buffer_size: 1000, recent_replay_ratio: 0.5`.
The first is a **static Folsom replay buffer built from Folsom's training split and handed to
every client**, adding replay samples equal to 50 % of each day's count. That is source-site raw
data resident on all three clients, which sits badly with the chapter's framing that raw
measurements never leave the site, and it is a second, undocumented channel through which Folsom
influences Poznań and Karpicko in *all* methods including Isolated. This must be either disclosed
in the Method section or switched off for the new campaign; it cannot stay silent.

**H-5. Attention is not the τ = 0.5 softmax the chapter describes.** `aggregate_global_prototypes`
uses an *adaptive* temperature (the standard deviation of the pairwise cosine-similarity matrix)
and adds a support-count prior (z-scored `log1p(count)`) to the logits. There is no τ parameter
anywhere in the code. `\Cref{eq:ch5-attention}` and "Experiments use τ = 0.5" describe a mechanism
that was not run. The Tier-3 "temperature τ = 0.1, 1.0" sweep has nothing to sweep until τ exists.

Two further mismatches are recorded in §5: the regime cut points are 0.30/0.65 in code versus
0.15/0.50 in the chapter, and the FedProx/Ditto/FedEMA hyper-parameters used were 1.0/1.0/0.1
rather than the 0.01/0.1 of `FL_BENCHMARK_EXPERIMENT_CARD.md`.

---

## 1. Repository map

### Entry points (`scripts/`)

| Script | Role for Chapter 5 |
|---|---|
| `scripts/fl_benchmark.py` | **The** Chapter 5 entry point. Click CLI, 1503 lines, contains the whole federated simulation loop. |
| `scripts/bash_scripts/run_fl_benchmark.sh` | The launcher/sweep used for the published runs. Hard-codes checkpoint and hyper-parameters; algorithms selected by commenting lines in a bash array. |
| `scripts/fl_edge_overhead_benchmark.py` | Payload and round-cost measurement (Table 5.5 source). Separate loop, not used by `fl_benchmark.py`. |
| `scripts/smart_persistence_model.py` | Smart Persistence baseline. Standalone; writes `.txt`/`.csv` into the current directory. |
| `scripts/clear_sky.py` | Generates `*_clear_sky.csv` per site with `pvlib` Ineichen + `lookup_linke_turbidity`. |
| `scripts/dvc/train_forecaster.py` | Source-model pre-training. |
| `scripts/edge_adaptation_benchmark.py`, `scripts/evaluate*.py`, `scripts/related_works/*` | Chapter 4 (single-device adaptation) paths. |

### Library (`cirr/`)

| Module | Contents relevant here |
|---|---|
| `cirr/fl/server.py` | `AGGREGATION_FNS` registry: `fedavg, fedbn, fedper, fedprox, ditto, fedema, fedproto, gated`; `is_bn_key`, `is_head_key`; weighted state-dict averaging. |
| `cirr/fl/kc_prototypes.py` | Regime labelling (`classify_irradiance`, `DEFAULT_KC_EDGES`), per-day prototype extraction, **attention aggregation** (`aggregate_global_prototypes`). |
| `cirr/fl/prototypes.py` | Generic binned extraction (`extract_binned_prototypes`), CSI-based extraction (`extract_prototypes`), accumulation, uniform aggregation (`aggregate_prototypes`). |
| `cirr/training/day_loop.py` | `predict_day`, `train_day` — the daily loop, the FedProx proximal term and **the prototype alignment loss**. |
| `cirr/models/forecaster.py` | `Forecaster` (timm encoder + 2-layer head), `forward_with_features`, `extract_proto_features`, freeze helpers, LoRA. |
| `cirr/models/loading.py` | `load_forecaster_from_checkpoint`. |
| `cirr/physics/clear_sky.py` | Clear-sky CSV lookup (values converted to kW/m²), minute-level fallback. |
| `cirr/physics/csi.py` | `compute_csi`, `discretize_csi` (3-bin edges 0.3/0.85; 5-bin option). Present but **not** on the FedProto path. |
| `cirr/data/forecasting.py`, `cirr/data/datasets/sky_image.py` | Dataset: sample = `(image, irr_history[4], ghi_clear, target, timestamp)`; irradiance divided by 1000 → kW/m². |
| `cirr/data/loading.py` | `load_dataset(path, split)`; image 128×128, ImageNet normalisation, `workers=0`. |
| `cirr/utils/dataset_utils.py` | `generate_year_days()` (365 MMDD keys, Feb 29 dropped), `group_indices_by_day()` (MMDD from the filename, year-invariant). |
| `cirr/utils/metrics.py` | `compute_metrics` → MAE, RMSE, WAPE, SMAPE, count. **No forecast skill, no per-regime split.** |
| `cirr/training/replay.py` | `RecentBuffer`, `build_folsom_replay_buffer`, `sample_replay_for_day`. |
| `cirr/continual/gating.py` | `build_gated_thresholds`, `train_gated_day` — the `gated` algorithm (WACV framework, not Chapter 5). |

### Configuration system

There is none in the sense the plan assumes. Configuration is **Click flags on
`fl_benchmark.py`**, wrapped by a bash launcher with shell variables. There is a `params.yaml`,
but it belongs to the DVC data-preparation and pre-training stages, not the FL benchmark. Every
`config.yaml`/`regime_label:`/`align_target:` key the plan asks for therefore has to be introduced,
and the cheapest route that keeps the existing interface is new Click options plus one resolved-config
dump written into the results directory.

### Results layout (current)

```
results/fl_benchmark/days-1_warmup30_sync7/<algorithm>/
  per_day.csv        day, day_num, fl_round_after, in_warmup, <client>_n/_mae/_rmse
  per_round.csv      fl_round, day, day_num, weight_change, days_in_round, warmup_round
  summary.json       overall_metrics per client + config echo
  fedproto_round_diagnostics.csv   (fedproto only) attention/support/edge per round
  checkpoints/<client>_final.pt, global_final.pt
```

`results/fl_benchmark/days-1_warmup30_sync7/` is the source of Table 5.3: its `overall_metrics`
MAE values × 1000 reproduce the table to the digit (Folsom static 34.93, isolated 31.93,
fedproto 31.66; Karpicko 77.45 / 50.34 / 50.21; Poznań 54.51 / 39.08 / 39.29).

---

## 2. Traced call chain — one "proposed method" run

```
scripts/bash_scripts/run_fl_benchmark.sh
  → python scripts/fl_benchmark.py --algorithm fedproto --checkpoint checkpoints/epoch_024.ckpt
       --dataset data/{folsom,karpicko,poznan}_prepared --sync-interval 7 --warmup-days 30 …

main()                                            fl_benchmark.py:1362
  ├ client names from --dataset basenames         fl_benchmark.py:1394-1398
  └ run_fl_benchmark(...)                         fl_benchmark.py:350

run_fl_benchmark
  ├ random/np/torch seeding                       fl_benchmark.py:448-450
  ├ load_dataset(path, split="test") ×3           cirr/data/loading.py:13 → ForecastingDataModule.setup
  │    └ SkyImageDataset.__getitem__              cirr/data/datasets/sky_image.py:57
  │         irradiance / 1000 → kW/m²; ghi_clear from *_clear_sky.csv (kW/m²)
  ├ group_indices_by_day(ds.periods) ×3           cirr/utils/dataset_utils.py:29   (MMDD key)
  ├ _load_checkpoint ×3 (identical start)         fl_benchmark.py:109 → cirr/models/loading.py
  ├ generate_year_days() → 365 MMDD, sliced       cirr/utils/dataset_utils.py:10
  ├ warmup cache probe (sha256 of hyper-params)   fl_benchmark.py:305,579-632
  └ for day_key in all_days:                      fl_benchmark.py:653
       ├ _process_client_day(...) per client      fl_benchmark.py:162   (ThreadPoolExecutor if --parallel-clients>1)
       │    ├ predict_day → preds, tgts, day_samples          cirr/training/day_loop.py:24
       │    ├ compute_metrics(preds, tgts)                     cirr/utils/metrics.py:77
       │    ├ train_day(..., global_prototypes, proto_lambda)  cirr/training/day_loop.py:73
       │    │    ├ AdamW rebuilt every day over requires_grad params   day_loop.py:135-140
       │    │    ├ forward_with_features → (mean, var, proto_feats)    forecaster.py:563
       │    │    ├ task loss: BerHu(mean*1000, tgt*1000)               day_loop.py:184-187
       │    │    └ ALIGNMENT: per-sample MSE(proto_feats[b], P[c(b)]),
       │    │         adaptive scale, + proto_lambda·scale·avg          day_loop.py:195-214
       │    ├ extract_day_prototypes  ← SECOND forward pass, model.eval()  kc_prototypes.py:67
       │    │    └ extract_binned_prototypes → model.extract_proto_features  prototypes.py:17
       │    └ recent_buffer.add_day                                     training/replay.py
       ├ accumulate_prototypes into _proto_accum[i]           fl_benchmark.py:727-733
       └ every sync_interval days (or at warmup end):
            aggregate_global_prototypes(_proto_accum)         cirr/fl/kc_prototypes.py:94
              → per-client personalised prototypes + attention matrix
            diagnostics rows appended                          fl_benchmark.py:854-886
            accumulators reset                                 fl_benchmark.py:889

after the loop
  ├ overall_metrics = compute_metrics(concat(all preds), concat(all tgts))   fl_benchmark.py:1069-1074
  └ main() writes per_day.csv / per_round.csv / fedproto_round_diagnostics.csv /
      checkpoints/*.pt / summary.json                                        fl_benchmark.py:1429-1499
```

Note the two forward passes per client-day: `predict_day` (evaluation, pre-training weights) and
`extract_day_prototypes` (post-training weights). See Q10.

---

## 3. Answers to Q1 – Q15

### Q1 — the alignment loss · **ANSWERED**

`cirr/training/day_loop.py:195-214`. For each sample *b* in the batch whose regime class
`c(b) = classify_irradiance(target_b, edges)` has a received prototype:

```
avg_proto = mean_b  MSE( proto_feats[b] , P[c(b)] )                 # element-wise mean squared error
scale     = clamp( task_loss.detach() / (avg_proto.detach() + 1e-8), max = 100.0 )
loss      = task_loss + proto_lambda · scale · avg_proto
```

* Tensors compared: `proto_feats[b]` — the stage-3 GAP embedding from `forward_with_features`
  (`forecaster.py:563`) — against the returned personalised prototype for that sample's class,
  detached and moved to device at `day_loop.py:131`.
* λ: `--fedproto-lambda`, default **0.1** (`fl_benchmark.py:1316-1318`), used as 0.1 in the
  published runs. Because of `scale`, λ is a *relative* weight: λ = 0.1 means the alignment term
  contributes ≈ 10 % of the task-loss magnitude, capped at 100× when embeddings approach the
  prototype.
* Classes without a received prototype contribute nothing (`if cls in proto_on_device`).
* The term is skipped entirely unless `global_prototypes is not None and proto_lambda > 0`
  (`day_loop.py:128`), so it is inactive during the 30-day warm-up and on the first post-warm-up day.

This closes the chapter's largest declared methodological gap (`\cref{eq:ch5-alignment}`). The
functional form is a scaled MSE, not a cosine term.

### Q2 — trainable scope · **ANSWERED (with a caveat)**

No method freezes anything. `_load_checkpoint` freezes only if `--freeze-early-backbone` is
passed (`fl_benchmark.py:109-114`), and the published runs did not pass it. `train_day` builds
`AdamW(filter(requires_grad, model.parameters()))` fresh **every day** for every method
(`day_loop.py:135-140`), so optimizer state does not persist across days — for Isolated, all five
baselines and the proposed method equally. Trainable parameter count is therefore identical
across methods at **22,426,609** (encoder 21,769,712 + head 656,897).

Caveat with consequences for the chapter: because the encoder is fully trainable and the
alignment term attaches to a stage-3 encoder activation, the gradient of the alignment term
flows into encoder stages 0–3. `\cref{fig:ch5-prototype-construction}` and
`\cref{fig:ch5-device-boundary}` both draw a *frozen* encoder. The figures are wrong, not the
code, and the chapter's own worry that "if the encoder is frozen … the do-no-harm result would
hold trivially" does not apply.

F2 as written ("make the trainable scope identical") is therefore already satisfied; what remains
is the *test* that prints and asserts the counts.

### Q3 — prototype layer and dimension · **ANSWERED**

`Forecaster.extract_proto_features` (`forecaster.py:532-561`) and the matching hook inside
`forward_with_features` (`forecaster.py:563-600`): if the encoder exposes `.stages` it hooks
`stages[3]`, takes `feat.mean(dim=(-2,-1))`; otherwise it returns the full pooled encoder output.
For `efficientvit_b2` at 128×128 the stage-3 output is `[B, 384, 4, 4]` → **d = 384**. For a
ResNet50 checkpoint the same call returns the pooled **2048**-d vector. Both branches were
verified by running the model. The pooled feature actually used for *prediction* is 2560-d, and
the head input is 2564.

### Q4 — prototype freshness and self-inclusion · **ANSWERED**

Fixed between rounds: **yes**. `_per_client_global_protos` is assigned only inside the
aggregation block (`fl_benchmark.py:821`) and passed unchanged into `train_day` on every
intervening day (`fl_benchmark.py:693-695`), detached at `day_loop.py:131`.

Self-inclusion: **yes**. In `aggregate_global_prototypes` (`kc_prototypes.py:129-143`) the
similarity matrix is `normed @ normed.T` over all clients *present* for that class, including the
diagonal, and `softmax(dim=1)` runs over that full row; `weighted = attn @ proto_stack` therefore
mixes the client's own prototype in with weight `attn[i,i]` — typically the largest weight, since
self-similarity is 1. If only one client has that class, it receives its own prototype back
verbatim with weight 1.0 (`kc_prototypes.py:122-127`), which makes the alignment term a
self-consistency penalty for that class rather than an exchange.

### Q5 — regime label and clear-sky availability · **ANSWERED, and it diverges from the chapter**

Labelling on the FedProto path: `classify_irradiance(_sample_target(sample), DEFAULT_KC_EDGES)`
(`kc_prototypes.py:27-34, 67-82`). `_sample_target` is `sample[3]`, the target irradiance already
divided by 1000 in the dataset, so the label variable is exactly the chapter's proxy
`k_c = GHI / 1000`. The **cut points are (0.30, 0.65)** (`kc_prototypes.py:23`), not the
(0.15, 0.50) the chapter states. A quantile-fitting helper `fit_kc_bin_edges` exists but is never
called on this path — `extract_day_prototypes` always uses the fixed defaults
(`kc_prototypes.py:74-75`), and the per-round "edge" diagnostics therefore log the constants.

Ineichen–Perez clear-sky **is already in the pipeline and reaches the sample**: generated by
`scripts/clear_sky.py` (pvlib `ineichen` + `lookup_linke_turbidity`), loaded by
`load_clear_sky_lookup` into kW/m² (`physics/clear_sky.py:10-30`), resolved per timestamp with a
minute-level fallback, and returned as `sample[2]`. Coverage is complete: 0 unresolved timestamps
at all three sites. A CSI-based classifier already exists too —
`prototypes.classify_csi_bin` → `compute_csi(target, ghi_clear)` → `discretize_csi` with 3-bin
edges 0.3/0.85 (`physics/csi.py`) — and is used by the `gated` algorithm, just not by `fedproto`.

F3 is consequently **wiring, not new physics**: swap the `classify_fn` passed to
`extract_binned_prototypes` and use the same function inside `train_day`. Note that the CSI edges
in `csi.py` (0.3/0.85) are clear-sky-index edges and are not interchangeable with the proxy edges
(0.30/0.65); the config switch must carry its edges with it.

### Q6 — daylight / zenith filter · **NONE EXISTS on this path**

No solar-zenith filter anywhere in `cirr/data`, `cirr/training`, `cirr/evaluation` or
`fl_benchmark.py`. `compute_metrics` filters only WAPE/SMAPE, via `min_threshold=0.005` kW/m²
(`metrics.py:77`); MAE and RMSE use every sample. A crude hour-window helper
`is_daylight_measurement` exists in `cirr/anomaly/embeddings.py:66` (fixed start/end hours) but is
used only by the anomaly pipeline and is not a zenith test.

Measured impact (see H-3): fraction of evaluated samples whose Ineichen–Perez clear-sky GHI is
exactly zero — Folsom 0.0 %, Karpicko 18.8 %, Poznań 30.8 %; with a 50 W/m² clear-sky cut
(≈ zenith 85°) the excluded fractions become ≈ 6.6 %, 27.2 %, 40.0 %.

Two consequences. First, F4 will change the Karpicko and Poznań numbers substantially and Folsom
barely, so the new table is not comparable to Table 5.3 row by row. Second, night samples all
carry `k_c < 0.30` and so land in the *overcast* prototype, meaning the Polish clients' overcast
prototype is currently dominated by dark frames while Folsom's is not — the regime labels are not
measuring the same thing across clients. This is a stronger version of the objection the chapter
already raises about latitude.

### Q7 — source checkpoint · **ANSWERED: the chapters differ**

Chapter 5 runs: `checkpoints/epoch_024.ckpt`, hard-coded at
`scripts/bash_scripts/run_fl_benchmark.sh:CHECKPOINT` and echoed in every `summary.json`.
Architecture `timm-efficientvit_b2`, loss BerHu, lr 3e-4, `lr_patience` 2, epoch 24,
global step 151,350, sha256 prefix `174130ae352871e2`.

Chapter 4 artefacts: `results/cl_benchmark/*/[site]_baseline_results.txt` records
`Checkpoint: checkpoints/epoch_010.ckpt`, which is `timm-resnet50`. Other ResNet50 checkpoints
present: `resnet50_epoch_015.ckpt` (BerHu), `epoch_007.ckpt` (GNLL, probabilistic).

So the two chapters use different checkpoints *and different architectures*. This fully explains
the chapter's own QUERY about Static rows disagreeing between Table 5.3 and Table 4.x. F5 is a
real blocker and needs a user decision (D-1).

### Q8 — normalized calendar and gaps · **ANSWERED**

`generate_year_days()` (`dataset_utils.py:10-26`) enumerates 366 dates of 2016 and drops Feb 29 →
365 `MMDD` keys. `group_indices_by_day()` (`dataset_utils.py:29-51`) sorts periods by filename and
keys them on characters 4:8 of the image name, i.e. MMDD with the year discarded; samples from
different years that share an MMDD merge into one day bucket. The simulation iterates the MMDD
list from `--start-day` (`fl_benchmark.py:494-501`).

Missing days are handled by omission, silently: `day_maps[i].get(day_key, [])` returns an empty
list, `predict_day` returns empty arrays, `train_day` returns 0.0 immediately, the day row shows
`---`, and **the day still counts toward `days_since_sync`**. A client with a long acquisition gap
therefore keeps consuming rounds while contributing nothing, and its stale prototypes simply do
not appear in `_proto_accum` — which is the natural-dropout case the plan's F8 wants to test
deliberately. Karpicko spans 2024-11 → 2025-10 and Poznań 2024-10 → 2025-07, so under the
normalized calendar Poznań has a structural hole from July to mid-October.

### Q9 — payload accounting · **ANSWERED (in the separate script only)**

`scripts/fl_edge_overhead_benchmark.py` measures **real serialized length**:
`_serialize_payload` writes through `io.BytesIO` + `torch.save` and reports `len(payload_bytes)`
(lines 137-143, 415-416, 427-429, 439-441). Uplink and downlink are tracked separately
(`upload_bytes_per_client`, `download_bytes_per_client`) and Table 5.5's "round-trip payload" is
their sum. Network time is `bytes·8 / (Mbps·1e6)` (line 153-156). A tensor-size helper
`_tensor_nbytes` also exists but is not what the reported figures use.

`scripts/fl_benchmark.py` measures **no bytes at all**. F12 is therefore "port the accounting from
the overhead script into the main loop", not "fix a formula".

Reconciliation of the chapter's numbers, now that d = 384 is known: prototypes 3 × 384 × 4 B =
4,608 B per direction, ≈ 9.2 kB + `torch.save` framing ≈ **0.013 MB** ✓. Weights 22,458,321
state-dict elements × 4 B = 89.8 MB per direction → 179.7 MB round trip analytically against
**171.676 MB** measured (−4.4 %, consistent with the download payload dropping non-float buffers).
The true value ratio is ≈ 1.3 × 10⁴ — i.e. the byte ratio and the value ratio agree, and the
chapter's "three orders of magnitude" understates it because it assumed 6,144 values instead of
1,152.

### Q10 — why a separate extraction pass · **ANSWERED**

Deliberate and, as currently sequenced, necessary. `_process_client_day` runs Step 1
`predict_day` (evaluation, **pre-update** weights) and only then Step 3 `train_day`; Step 4
extracts prototypes afterwards precisely so that "features reflect the updated model"
(`fl_benchmark.py:265-275`). The embeddings produced during `predict_day` belong to the old
weights, and those produced inside `train_day` belong to intermediate weights mid-epoch, so
neither can stand in for a post-update prototype.

`fig:ch5-prototype-construction`'s claim that "the embeddings come from the forward pass the
forecaster already performs" is therefore not true of the implementation. F13's precondition is
not met: the pass can only be removed by accepting *mid-epoch* embeddings (bias) or by moving
extraction before training (different semantics). Recommendation: document it, keep the pass, and
reuse the already-loaded `day_samples` tensors — which the code already does, so the cost is
forward compute only, not I/O.

### Q11 — the 4.4 s training-time difference · **PARTIALLY OPEN**

The difference is in `scripts/fl_edge_overhead_benchmark.py`, not in `fl_benchmark.py`, and the
overhead script runs its own loop (640 samples, batch 32, 3 rounds). Reading the two branches,
the prototype configuration and the weight configuration do execute the same local training call,
so no code path explains a systematic 4.4 s gap; with a single measurement per configuration and
no repetition, measurement noise on a Raspberry Pi 5 is the most economical explanation. **OPEN**
until the device plan re-measures with repeats. Nothing in Phase 1 to 4 depends on it, but the
chapter should not assert the difference as a finding until it is reproduced — it currently
carries a QUERY doing exactly that.

### Q12 — determinism · **PARTIAL**

Set: `random.seed`, `np.random.seed`, `torch.manual_seed` at `fl_benchmark.py:448-450`; per
client-day derived seeds `seed + day_num·n_clients + i` (`fl_benchmark.py:679`) feeding a local
`random.Random` for batch shuffling (`day_loop.py:144-146`) and replay sampling.

Not set: `torch.cuda.manual_seed_all`, `torch.backends.cudnn.benchmark = False`,
`torch.use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG`. DataLoader seeding is moot —
`load_dataset` uses `workers=0` and the loops index the dataset directly rather than using a
`DataLoader`. One genuine nondeterminism risk beyond kernel-level: with `--parallel-clients > 1`
clients run in a `ThreadPoolExecutor` on separate CUDA streams (`fl_benchmark.py:643-648,
204-208`); per-client state is disjoint so results should be stable, but this has never been
tested. The published runs used `--parallel-clients 3`.

F10 is a small, real change. The determinism check S3 should be run *with* `--parallel-clients 3`,
since that is the configuration the campaign will use.

### Q13 — client participation / dropout · **NOT SUPPORTED**

No `dropout`, `participation`, `outage` or straggler concept exists anywhere in `cirr/` or
`scripts/`. The only exclusion mechanism is `--loso-holdout SITE` (`fl_benchmark.py:463-464,
817-820, 942-944, 958-960`), which removes one named client from *every* round for the whole run —
permanent, not per-round. It is a good structural template for F8: the aggregation and
distribution loops already know how to skip a client on both sides, so F8 becomes "make the skip
set a per-round function" rather than new plumbing. Natural gaps additionally exist for free (Q8).

### Q14 — mid-run client join · **NOT SUPPORTED**

No join-day or history-window concept. All clients are constructed up front from `--dataset`
flags, all load the same checkpoint, and all participate from day 0. F9 needs: a per-client
`join_day` that suppresses training, evaluation contribution and round participation before that
day, plus a `local_history_days` window that restricts which prior days a joining client may train
on. The day loop is index-driven per client, so this is contained, but it is the largest of the
new features and it interacts with the warm-up cache (a joining client has no warm-up state).

### Q15 — Smart Persistence · **EXISTS, BUT DISCONNECTED AND IN DIFFERENT UNITS**

`scripts/smart_persistence_model.py` implements it (clear-sky-index persistence with naive-persistence
fallback when clear-sky data is missing) and per-site results already exist:
`results/cl_benchmark/{folsom,karpicko,poznan}/*_smart_persistence_15min_results.txt` plus
matching per-day tables and prediction CSVs. Karpicko for example reports overall MAE 54.8746 and
RMSE 127.2835 over the same 226,114 samples, with 42,554 naive fallbacks.

Three obstacles before FS can be filled:
1. **Units.** Smart-persistence outputs are in W/m², model results in kW/m² — the scripts do not
   share a scaling convention. Mixing them without conversion silently produces a 1000× error.
2. **Not logged per FL run.** `fl_benchmark.py` never computes it, so `sp_rmse` in the plan's
   `metrics_daily.csv` has to be joined in or recomputed inside the run.
3. **Population.** The existing runs cover all samples, including night; once F4 lands the
   Smart-Persistence reference must be recomputed on the *filtered* population or FS will be
   computed against a different denominator than the model rows.

The good news for the chapter's QUERY on the empty FS column: the raw material exists, and the
column can be filled from the existing artefacts for the *current* (unfiltered) table without any
new GPU time.

---

## 4. Verification of the thesis description (plan §2.2)

| Item | Thesis says | Code / artefact says | Verdict |
|---|---|---|---|
| Task | 15-min-ahead GHI regression | horizon fixed in the prepared periods; 15 min | ✅ |
| Encoder | ResNet50, pooled 2048-d | `timm-efficientvit_b2`, pooled 2560-d | ❌ **H-1** |
| Fusion / head | 2052 → 256 → 1 | 2564 → 256 → 1 | ❌ |
| Scalars | `I_t, I_{t-5}, I_{t-10}, I_{t-15}` | 4 history values, /1000 | ✅ |
| Source model recipe | AdamW 3e-4, patience-5 halving, batch 128, ≤30 ep, patience 10, 20 % day split | ckpt hparams lr 3e-4, `lr_patience` 2, epoch 24; batch/epochs not recorded in the ckpt | ⚠️ partly unverifiable |
| Loss | not stated | BerHu on ×1000 scaled values (`day_loop.py:184-186`) | ⚠️ undocumented |
| Clients / periods | Folsom 362 d / 246,164; Poznań 204 d / 183,044; Karpicko 301 d / 226,114 | exactly these counts | ✅ |
| Normalized calendar | year ignored | MMDD keying, Feb 29 dropped | ✅ |
| Client training | AdamW 1e-5, wd 1e-4, clip 1.0, batch 32, 1 epoch/day | identical in `summary.json` | ✅ |
| Schedule | 30-day warm-up, sync every 7 d | `warmup_days 30`, `sync_interval 7`, 48 rounds | ✅ |
| **Replay** | not mentioned | Folsom reservoir 10,000 @ ratio 0.5 **+** recent FIFO 1,000 @ ratio 0.5, on every client, every method | ❌ **H-4** |
| Regimes | `k_c` = GHI/1000, cuts 0.15 / 0.50 | same variable, cuts **0.30 / 0.65** | ❌ |
| Prototype dim | `d = 2048`, payload 6,144 values | 384, payload 1,152 values | ❌ **H-2** |
| Attention | softmax cosine, τ = 0.5 | adaptive temperature (std of sim matrix) **+** log-count prior; no τ | ❌ **H-5** |
| Alignment loss | "not yet stated" | scaled MSE, λ = 0.1 (Q1) | ✅ now resolved |
| Encoder frozen (figures) | frozen | fully trainable | ❌ |
| Baselines | FedAvg, FedProx, FedBN, FedPer, FedEMA | all present, plus Ditto, FedProto, gated | ✅ |
| FedProx μ / Ditto λ / FedEMA α | card says 0.01 / 0.1 / — | runs used **1.0 / 1.0 / 0.1** | ⚠️ |
| Metrics | MAE, RMSE, WAPE, SMAPE, FS | first four yes; **FS absent** | ⚠️ |
| Evaluation population | "all samples, no minimum-irradiance threshold" | true — and that is the problem (H-3) | ⚠️ |
| FedBN/FedPer payload | Table 5.1 implies reduced payload | both upload full state dicts; only server-side keys are dropped (`fl_benchmark.py:44-49`) | ❌ |

---

## 5. Gap table (plan §§4–7)

Effort: S < 2 h, M < 1 day, L > 1 day.

### Phase 1 fixes

| Plan item | Status | Evidence | Proposed change | Effort | Risk / dependency |
|---|---|---|---|---|---|
| F1 alignment loss + λ | **exists** (documented above) | `day_loop.py:195-214`; `fl_benchmark.py:1316` | No code change; record the form in this document and in the chapter. Optionally expose the `scale` cap. | S | The adaptive `scale` means λ is not comparable to a plain FedProto λ; Tier-3 "λ at 0.25× / 4×" sweeps the *relative* weight. |
| F2 identical trainable scope | **exists**; test **new** | `day_loop.py:135-140`; no freeze by default | Add one test asserting equal trainable counts across methods; print per method. | S | None. Chapter figures must be corrected instead (encoder is trainable). |
| F3 clear-sky regime label | **modify** | `kc_prototypes.py:23,67-82`; `prototypes.py:13-14`; `csi.py` | Add `--regime-label proxy\|clear_sky` + explicit edges; route the chosen `classify_fn` into both `extract_day_prototypes` **and** `train_day`'s in-loop labelling (`day_loop.py:199`), which currently re-derives the label independently. | M | Both call sites must use the same labeller or the alignment target silently mismatches the prototype. Default `clear_sky` changes all numbers. |
| F4 zenith < 85° filter | **new** | no filter exists (Q6) | One predicate, applied where `day_indices` are built, shared by train and eval; derive from `ghi_clear` (already per sample) or add pvlib zenith to the clear-sky CSV. | M | **Changes every published number** (H-3). Requires D-2. Chapter 4 comparability requires the same filter there. |
| F5 one shared checkpoint | **blocked** | ch5 `epoch_024` (EfficientViT) vs ch4 `epoch_010` (ResNet50) | Pick one checkpoint for both chapters; log path + sha256 in `config.yaml`. | S (logging) / L (if retraining) | **D-1.** Choosing ResNet50 makes the chapter text correct but invalidates every existing Chapter 5 number; choosing EfficientViT requires rewriting the Method paragraph and both figures. |
| F6 `align_target` controls | **new** | `fl_benchmark.py:821`, `kc_prototypes.py:94` | Post-process the aggregation output: `own` = skip exchange (and log zero bytes), `shuffled` = permute class keys of the received dict per round, `random` = Gaussian with matched per-class norm. | M | `own` is not a no-op: the alignment term still fires against the client's own prototype, which is the H5.2 triviality check the plan wants. Seed the permutation from the run seed. |
| F7 `attention` / `stratify` switches | **new** | `kc_prototypes.py:94-150` | `uniform` → reuse existing `aggregate_prototypes` (`prototypes.py:76`); `single` → one class for all samples. | S | `single` collapses `C` to 1, so payload drops to `d` values — the ablation moves the communication number too; report both. |
| F8 participation schedules | **new** | only `--loso-holdout` exists (Q13) | Generalise the holdout predicate to a per-round `participating(client, round)`; absent clients neither contribute nor receive; log `participated`. | M | Interacts with natural gaps (Q8): a client may be "participating" yet have no data. Log both flags separately. |
| F9 `join_day` / `local_history_days` | **new** | none (Q14) | Per-client join day gating training, metric accumulation and round participation; history window restricting trainable prior days. | L | Largest new feature; interacts with the warm-up cache (joining clients have no warm-up state) — simplest is to disable the cache for join runs. |
| F10 determinism | **modify** | `fl_benchmark.py:448-450`; nothing else (Q12) | Add the plan's snippet; verify under `--parallel-clients 3`. | S | `use_deterministic_algorithms(True)` may fail on some timm/EfficientViT kernels; use `warn_only=True` as the plan suggests. |
| F11 logging schema | **new** | current outputs are `per_day.csv` / `per_round.csv` / `summary.json` only | Write the six files of §5.4; keep the existing three for backward compatibility. Long-format `metrics_daily.csv` replaces the current wide `<client>_mae` columns. | M | `metrics_regime.csv` needs per-sample regime labels retained through `predict_day`, which currently discards them. `attention.npy` needs a fixed `rounds × clients × clients × regimes` array — present classes vary per round, so absent entries must be NaN, not 0. |
| F12 serialized byte counts | **modify** | `fl_edge_overhead_benchmark.py:137-143` has it; `fl_benchmark.py` has nothing (Q9) | Import/lift `_serialize_payload` into a shared helper; call it at each round for the actual uploaded and downloaded objects. | S | For FedBN/FedPer the honest number is the full state dict (they upload everything, `fl_benchmark.py:44-49`) — log what is actually sent and note it. |
| F13 reuse embeddings | **blocked → document** | `fl_benchmark.py:265-275`; Q10 | Do not remove the pass. Document why (post-update semantics) and correct the chapter figure. | S | Removing it changes prototype semantics; this is a chapter-text fix, not a code fix. |
| F14 pooled oracle | **new** | no pooled mode | New algorithm `pooled`: one model, daily union of all clients' day indices, evaluated per client through the same `predict_day`. | M | Union day is ~3× the samples → ~3× the daily training time; a full-year pooled run is the most expensive single run in the plan. |
| F15 FedAvg+FT / FedProto / Ditto | **mostly exists** | `AGGREGATION_FNS` (`server.py:385-394`) has `ditto` and `fedproto`; `ditto` already runs a dual model (`fl_benchmark.py:1008-1060`) | Only "FedAvg + local fine-tuning" is new: FedAvg aggregation plus N local steps after distribution. | S | Ditto is present but was not in the published table; including it widens the baseline set beyond the chapter's five. |

### Phase 2 (smoke) items

| Plan item | Status | Evidence | Proposed change | Effort | Risk |
|---|---|---|---|---|---|
| §5.1 window selection, `configs/smoke_window.yaml` | **new** | no `configs/` directory | Script the 80 %-daylight-coverage scan over the normalized calendar; emit the manifest. | S | "Expected daylight samples" needs a definition — propose: median daylight sample count for that site across ±15 days. |
| §5.2 shortened schedule | **exists** | `--warmup-days`, `--sync-interval`, `--num-days`, `--start-day` | Flags only. | S | Warm-up cache key includes `warmup_days`, so smoke runs get their own cache entry — fine. |
| §5.3 configuration matrix | **partly** | see F6–F9, F14, F15 | Depends on Phase 1. | — | Regression row (S10) needs the pre-change code; use `git stash`/worktree at the Phase-0 commit. |
| §5.4 logging schema | see F11 | — | — | M | — |
| S1–S11 acceptance checks | **new** | — | One checker script reading the smoke outputs. | M | S2's "daily MAE below 300 W/m²" must account for the kW/m² internal scale (×1000). |
| S10 regression to pre-change code | **new** | — | Run old settings on the same 7 days and diff. | S | Cannot hold to 1e-4 if F4 is on; run the regression with the filter *off* by construction. |

### Phase 3 (full-year) items

| Plan item | Status | Evidence | Proposed change | Effort | Risk |
|---|---|---|---|---|---|
| Runner with resume + parallel jobs + `failures.log` | **new** (a launcher exists but is not resumable) | `scripts/bash_scripts/run_fl_benchmark.sh` — algorithms selected by commenting lines, no skip-if-exists, no failure log | Short Python runner over a config list; skip when `summary.json` exists. | S | — |
| Seeds 3–5 per configuration | **exists** | `--seed` | — | S | Warm-up cache key includes the seed, so each seed pays its own 30-day warm-up. |
| Tier 1 / 2 / 3 matrices | **depends** | — | — | — | See D-4 on hardware and D-5 on scale. |

### Phase 4 (analysis) items

| Plan item | Status | Evidence | Proposed change | Effort | Risk |
|---|---|---|---|---|---|
| Forecast skill as primary metric | **blocked on inputs** | Q15 | Recompute Smart Persistence per site on the filtered population, in consistent units, and log `sp_rmse` daily. | M | **D-3.** Without it FS is unavailable and MAE becomes primary by default. |
| Moving-block bootstrap, TOST | **new** | — | numpy/scipy only. | M | — |
| Diebold–Mariano + Newey–West + Holm | **new** | `statsmodels` **is not installed** | ~20 lines with numpy/scipy (NW variance and Holm are short), or install `statsmodels`. | S–M | **D-6** (new dependency; rule 5). Recommendation: hand-roll, no new dependency. |
| Per-regime tables | **depends on F11** | regime labels not currently retained | — | M | — |
| Cold-start / late-join analysis | **depends on F9** | — | — | M | — |
| `tables/`, `figures/`, `ANALYSIS_REPORT.md` | **new** | — | — | M | — |

---

## 6. Plan items that do not fit the current software

1. **F13 (reuse the daily forward pass).** Its premise is false; the extraction pass exists for a
   reason (Q10). → Convert to a documentation fix plus a chapter-figure correction.
2. **Tier 3 "temperature τ = 0.1, 1.0".** There is no τ (H-5). → Either add a fixed-τ softmax as an
   *alternative* aggregation mode and sweep it, or replace the sweep with "adaptive vs fixed
   temperature" as a two-level comparison. The second is cheaper and answers the same question.
3. **Tier 3 "C = 2, C = 4 from clear-sky-index terciles".** `discretize_csi` supports only 3 and 5
   bins (`csi.py:19-39`) and `PROTO_CLASS_NAMES` hard-codes three names. → Generalise the labeller
   to arbitrary edge lists; then C = 2, 3, 4, 5 all follow.
4. **"Config key `regime_label:` …" and `config.yaml`.** There is no YAML config layer (§1). →
   Keep Click as the interface, and dump the fully resolved configuration (plus git commit and
   checkpoint sha256) to `config.yaml` in the results directory. This satisfies every downstream
   consumer without building a config framework.
5. **Plan §2.2's model description.** Not a software gap but a plan gap: the plan's own context
   table reproduces the chapter's ResNet50/2048 description, which does not match the code. It
   should be corrected in the plan once D-1 is settled, otherwise Phase 1 will be validated
   against the wrong target.
6. **FedBN/FedPer as "low / medium bandwidth" in Table 5.1.** The implementation uploads full
   state dicts for both (`fl_benchmark.py:44-49`). Either implement key-filtered upload or mark
   the table as analytical-not-measured for those rows.

---

## 7. Decisions needed from the user

**D-1 — Checkpoint and architecture (blocks F5, and everything downstream).**
Chapter 5 ran on EfficientViT-B2 (`epoch_024.ckpt`); Chapter 4 ran on ResNet50 (`epoch_010.ckpt`).
Options: (a) standardise on EfficientViT-B2 and rewrite the Chapter 5 Method paragraph and both
figures (cheapest, keeps existing results as a reference point); (b) standardise on a ResNet50
checkpoint, making the chapter text correct but requiring the entire campaign to be rerun and all
published numbers discarded; (c) keep both and state per-chapter checkpoints explicitly. My
recommendation is (a), with the chapter text corrected — the encoder identity is not a claim the
chapter's argument depends on, whereas cross-chapter comparability is.

**D-2 — The replay buffers (H-4).** The published runs gave every client a 10,000-sample Folsom
replay buffer at ratio 0.5 plus a 1,000-sample recent FIFO at ratio 0.5. Options: (a) disable both
for the new campaign (cleanest federated story, all numbers change); (b) keep them and document
them in the Method section, including the fact that Folsom raw samples are resident on every
client; (c) keep only the recent FIFO, which is local data and raises no cross-site issue. I
recommend (c) as the default arm with (a) as a Tier-3 sensitivity row.

**D-3 — Forecast skill.** Filling the FS column requires Smart Persistence recomputed per site on
the same filtered population and unit convention (Q15). Confirm that FS should be the primary
comparison metric as §7.1 states — if so, this is a prerequisite for Tier 1 analysis, not an
optional extra.

**D-4 — Which machine is "the server".** This machine has an **RTX PRO 500 Blackwell laptop GPU
with 6 GiB** and all 147 GB of prepared data present locally. The plan budgets 80–160 GPU-hours
for Tier 1 alone and assumes parallel jobs. Six GiB will not hold many concurrent runs of a 22 M
parameter model at batch 32. Confirm whether the campaign runs here or elsewhere; if elsewhere,
confirm the data is staged there.

**D-5 — Tier scope.** Tiers 1–3 total ≈ 175 run-equivalents. No existing artefact records wall
time, so the 1–2 GPU-hour estimate is unverified (see §8). I propose measuring one full-year run
at the end of Phase 2 and re-scoping the tiers against the measurement before Checkpoint C.

**D-6 — `statsmodels`.** Not installed. Diebold–Mariano with a Newey–West variance and Holm
correction is ~20 lines on numpy/scipy. Approve hand-rolling (recommended) or approve the
dependency.

**D-7 — What happens to the existing Chapter 5 numbers.** F4 alone will move Karpicko and Poznań
materially while leaving Folsom nearly unchanged (H-3), so the new table will not be row-comparable
with Table 5.3. Confirm that Table 5.3 is to be *replaced* by the new campaign rather than
supplemented, and that the sub-1 % do-no-harm margins are expected to be re-derived from scratch.

---

## 8. Runtime estimate

**Not derivable from existing logs.** No artefact records wall time: `summary.json` has no timing
field, and the per-run `checkpoints/` and `summary.json` mtimes for the eight published algorithms
cluster between 21:30 and 21:54 on 2026-06-09, which is consistent with either a parallel sweep or
a fast sequential one and distinguishes neither. Two further confounders: the published runs loaded
a warm-up cache (`warmup_cache_loaded: true`, saving 30 days each) and the `warmup_cache/`
directory no longer exists, so the first new run pays the warm-up back.

Scale of one full-year run: 365 days × 3 clients, ≈ 655k evaluated samples per client-year, one
forward pass for evaluation, one training epoch, plus — for prototype methods — a second forward
pass for extraction, plus replay samples at whatever ratio D-2 settles on. `summary.json` reports
48 aggregation rounds for a 365-day run at sync 7 with a 30-day warm-up.

**Proposal:** add wall-clock timing to `summary.json` in Phase 1 (F11 already asks for
`wall_time_s`), measure a full-year Isolated run and a full-year proposed run once at the end of
Phase 2, and put the measured extrapolation into `SMOKE_REPORT.md` (check S11) before any tier is
launched.

---

## 9. What Phase 0 did not change

No source file was modified. Two files were added on branch `ch5-benchmarks`:
`docs/ch5_server_benchmark_plan.md` (the plan itself) and this report. No result directory was
read destructively or written.
