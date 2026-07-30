# Available Trained Configurations — `/glade/work/tsalin/CMIP`

This file is the authoritative inventory of which **(setup, sample size,
architecture, alignment method, variable, hyperparameters, split)**
combinations have already been trained and have their outputs — model
checkpoint, `quality_df*.pkl`, `*_latent_representations_df.pkl` — available
under `/glade/work/tsalin/CMIP/model_evaluation/` and
`/glade/work/tsalin/CMIP/latent_representations/` (see the main
[README](README.md), §3, for what each of those folders contains).

**229 configurations** are currently recorded below, all with status
**Available**. As new runs are submitted (`jobs/run_*.pbs`), add a row here —
use **In progress** while a job is queued/running and **Missing** for a
configuration that was planned but failed / hasn't been launched yet, so this
file stays a reliable map of what a notebook can actually load without
re-running a job first.

### How to read the columns

| Column | Meaning |
|---|---|
| **Setup** | Matches a `model_evaluation/<Setup>/` folder and a training notebook family (see README §4). For exp3, the sub-variant (`AEh`, `AEhs2`, `AEhs2s3`, `AEall`) refers to the autoencoder architecture depth/skip-connections. |
| **Samples** | `NUM_SAMPLE` job argument — number of patches used, drawn from the matching `derived/multivariate_samples*_v<N>` folder on scratch. Matches the `ns<N>` token in output filenames. |
| **AE type** | `AUTOENCODER_TYPE` — encoder/decoder backbone: `CNN` or `MLP`. |
| **Align. method** | `ALIGNMENT_METHOD` — `swd` (Sliced-Wasserstein Distance), `swdn` (normalized SWD), or `adversarial` (classifier-based alignment). Blank where no alignment term applies (e.g. plain baselines). |
| **Variable** | `VARIABLE` job argument — target predicted from the latent (`pr`, `tas`...), or `central_6` for the masked-point task (predicting the 6 central grid points of the patch, see README §4 "mask" notebooks). |
| **λ_pred** | `LAMBDA_PRED` — weight of the prediction loss term. |
| **λ_align** | `LAMBDA_ALIGN` — weight of the alignment loss term. |
| **Split** | Train / validation / test fractions (`VAL_FRACTION`, `TEST_FRACTION` job arguments; train = 1 − val − test). |

Blank cells (`–`) mean the value wasn't recorded for that run — either the
parameter doesn't apply (e.g. no alignment method for a baseline) or the run
predates that hyperparameter being tracked explicitly.

---

## `pr` — 100,000 samples (λ_pred = 0.676, λ_align = 0.054, split 60/15/25)

| Status | Setup | AE type | Align. method |
|---|---|---|---|
| Available | Exp3 / AEh | CNN | swd |
| Available | Exp3 / AEhs2 | CNN | swd |
| Available | Exp3 / AEhs2s3 | CNN | swd |
| Available | Exp3 / AEall | CNN | swd |
| Available | Exp4 | CNN | swd |
| Available | Exp5 / BS | CNN | swd |
| Available | Exp5 / CERA noalign | CNN | swd |
| Available | Exp5 / CERA | CNN | swd |

## `pr` — 1,000,000 samples (λ_pred = 0.676, λ_align = 0.054, split 60/15/25)

| Status | Setup | AE type | Align. method |
|---|---|---|---|
| Available | Exp3 / AEh | CNN | swd |
| Available | Exp4 | CNN | swd |
| Available | Exp5 / BS | CNN | swd |
| Available | Exp5 / CERA noalign | CNN | swd |
| Available | Exp5 / CERA | CNN | swd |

## `pr` — 10,000 samples (λ_pred = 0.01, λ_align = 0.0001, split 70/10/20)

| Status | Setup | AE type | Align. method |
|---|---|---|---|
| Available | Exp3 / AEh | CNN | swd |
| Available | Exp3 / AEhs2 | CNN | swd |
| Available | Exp3 / AEhs2s3 | CNN | swd |
| Available | Exp3 / AEall | CNN | swd |
| Available | Exp4 | CNN | swd |
| Available | Exp5 / BS | CNN | swd |
| Available | Exp5 / CERA noalign | CNN | swd |
| Available | Exp5 / ClimaX | CNN | swd |
| Available | Exp5 / physical | CNN | swd |
| Available | Exp5 / full latent | CNN | swd |
| Available | Exp5 / CERA | CNN | swd |

## `pr` — 1,000,000 samples (λ_pred = 0.01, λ_align = 0.0001, split 80/5/15)

| Status | Setup | AE type | Align. method |
|---|---|---|---|
| Available | Exp3 / AEh | CNN | swd |
| Available | Exp3 / AEhs2 | CNN | swd |
| Available | Exp3 / AEhs2s3 | CNN | swd |
| Available | Exp3 / AEall | CNN | swd |
| Available | Exp4 | CNN | swd |
| Available | Exp5 / BS | CNN | swd |
| Available | Exp5 / CERA noalign | CNN | swd |
| Available | Exp5 / ClimaX | CNN | swd |
| Available | Exp5 / physical | CNN | swd |
| Available | Exp5 / full latent | CNN | swd |
| Available | Exp5 / CERA | CNN | swd |

## `tas` — 1,000,000 samples (λ_pred = 0.01, λ_align = 0.0001, split 80/5/15)

| Status | Setup | AE type | Align. method |
|---|---|---|---|
| Available | Exp3 / AEh | CNN | swd |
| Available | Exp3 / AEhs2 | CNN | swd |
| Available | Exp3 / AEhs2s3 | CNN | swd |
| Available | Exp3 / AEall | CNN | swd |
| Available | Exp5 / BS | CNN | swd |
| Available | Exp5 / CERA noalign | CNN | swd |
| Available | Exp5 / ClimaX | CNN | swd |
| Available | Exp5 / physical | CNN | swd |
| Available | Exp5 / full latent | CNN | swd |
| Available | Exp5 / CERA | CNN | swd |

## `central_6` (masked-point task) — 1,000,000 samples (λ_pred = 0.01, λ_align = 0.0001, split 80/5/15)

| Status | Setup | AE type | Align. method |
|---|---|---|---|
| Available | Exp5 / CERA noalign | CNN | swd |
| Available | Exp5 / BS | CNN | swd |
| Available | Exp5 / CERA | CNN | swd |

## `pr` — 1,000,000 samples, Exp5 / CERA — extra single runs (λ_pred = 0.01, λ_align = 0.0001, split 80/5/15)

| Status | AE type | Align. method |
|---|---|---|
| Available | CNN | adversarial |
| Available | MLP | swd |

## `pr` — 1,000,000 samples, Exp5 / CERA — (λ_pred, λ_align) grid, split 80/5/15

The same 22-point hyperparameter grid below is available for **all 5** of the
following encoder / alignment combinations: `CNN + swd`, `CNN + swdn`,
`CNN + adversarial`, `MLP + swd`, `MLP + swdn` — i.e. 110 trained runs in
total for this block.

| λ_pred | λ_align |
|---|---|
| 0.1 | 0.0001 |
| 0.01 | 0.001 |
| 0.1 | 0.001 |
| 0.01 | 0.01 |
| 0.1 | 0.01 |
| 0.01 | 0.1 |
| 0.1 | 0.1 |
| 0.3 | 0.3 |
| 0.1 | 0.25 |
| 0.1 | 0.45 |
| 0.1 | 0.65 |
| 0.1 | 0.85 |
| 0.25 | 0.1 |
| 0.25 | 0.25 |
| 0.25 | 0.45 |
| 0.25 | 0.65 |
| 0.45 | 0.1 |
| 0.45 | 0.25 |
| 0.45 | 0.45 |
| 0.65 | 0.1 |
| 0.65 | 0.25 |
| 0.85 | 0.1 |

## `pr` — 2,500,000 samples (split 80/5/15)

| Status | Setup | AE type | Align. method | λ_pred | λ_align |
|---|---|---|---|---|---|
| Available | Exp3 / AEh | CNN | – | – | – |
| Available | Exp3 / AEhs2 | CNN | – | – | – |
| Available | Exp3 / AEhs2s3 | CNN | – | – | – |
| Available | Exp3 / AEall | CNN | – | – | – |
| Available | Exp4 | CNN | swd | – | 0.45 |
| Available | Exp5 / BS | – | – | – | – |
| Available | Exp5 / CERA noalign | CNN | – | 0.1 | – |
| Available | Exp5 / ClimaX | – | – | – | – |
| Available | Exp5 / physical | – | – | – | – |
| Available | Exp5 / full latent | CNN | swd | 0.1 | 0.45 |
| Available | Exp5 / CERA | CNN | swd | 0.1 | 0.45 |
| Available | Exp5 / CERA | CNN | swdn | 0.1 | 0.45 |
| Available | Exp5 / CERA | CNN | adversarial | 0.1 | 0.45 |
| Available | Exp5 / CERA | CNN | swdn | 0.1 | 0.85 |

## `pr` — 1,000,000 samples, Exp5 / CERA seasonal — (λ_pred, λ_align) grid, split 80/5/15

CNN encoder throughout.

**Align. method = swd**

| λ_pred | λ_align |
|---|---|
| 0.45 | 0.1 |
| 0.1 | 0.45 |
| 0.1 | 0.1 |
| 0.1 | 0.25 |
| 0.1 | 0.65 |
| 0.1 | 0.85 |
| 0.25 | 0.1 |
| 0.25 | 0.25 |
| 0.25 | 0.45 |
| 0.25 | 0.65 |
| 0.45 | 0.25 |
| 0.45 | 0.45 |
| 0.65 | 0.1 |
| 0.65 | 0.25 |
| 0.85 | 0.1 |
| 0.3 | 0.3 |

**Align. method = swdn**

| λ_pred | λ_align |
|---|---|
| 0.85 | 0.1 |
| 0.1 | 0.85 |
| 0.1 | 0.1 |
| 0.3 | 0.3 |
| 0.1 | 0.25 |
| 0.1 | 0.45 |
| 0.1 | 0.65 |
| 0.25 | 0.1 |
| 0.25 | 0.25 |
| 0.25 | 0.45 |
| 0.25 | 0.65 |
| 0.45 | 0.1 |
| 0.45 | 0.25 |
| 0.45 | 0.45 |
| 0.65 | 0.1 |
| 0.65 | 0.25 |

## `pr` — 1,000,000 samples, Exp5 / CERA full latent — (λ_pred, λ_align) grid, split 80/5/15

CNN encoder, align. method = swd throughout (alignment applied to the full
64-dim latent instead of 48, see README §1).

| λ_pred | λ_align |
|---|---|
| 0.1 | 0.1 |
| 0.3 | 0.3 |
| 0.1 | 0.25 |
| 0.1 | 0.45 |
| 0.1 | 0.65 |
| 0.1 | 0.85 |
| 0.25 | 0.1 |
| 0.25 | 0.25 |
| 0.25 | 0.45 |
| 0.25 | 0.65 |
| 0.45 | 0.1 |
| 0.45 | 0.25 |
| 0.45 | 0.45 |
| 0.65 | 0.1 |
| 0.65 | 0.25 |
| 0.85 | 0.1 |

## `pr` — 1,000,000 samples, Exp5 / CERA No align — λ_pred sweep, split 80/5/15

CNN encoder, no alignment term.

| Status | λ_pred |
|---|---|
| Available | 0.1 |
| Available | 0.3 |
| Available | 0.25 |
| Available | 0.45 |
| Available | 0.65 |
| Available | 0.85 |

## `pr` — 1,000,000 samples, Exp5 / ClimaX - boosted (split 80/5/15)

| Status | Setup |
|---|---|
| Available | Exp5 / ClimaX - boosted |

---

*Last updated: 2026-07-30. Maintained by hand — update this file whenever a
new `jobs/run_*.pbs` job finishes (or fails) so it keeps matching what's
actually on `/glade/work/tsalin/CMIP/`.*
