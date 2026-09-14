# Climate-Invariant Representation Learning

**Research project — Gentine Lab, Columbia University · New York, USA · April–August 2026**

This repository contains the code and analyses developed during my research internship at **Columbia University's Gentine Lab**, investigating whether climate-invariant latent representations can improve **out-of-distribution (OOD) generalization under climate change**.

Using **CESM2 simulations from CMIP6**, I studied representation alignment across historical and future climate scenarios, developed a normalized Sliced Wasserstein alignment objective for **CERA**, and compared its OOD behavior with the **ClimaX** climate foundation model.

> **Main result:** the proposed normalized alignment objective reduced the degradation in precipitation prediction performance by approximately **80%** from Historical to SSP5-8.5 compared with ClimaX (**R² drop: 0.04 vs. 0.21**).

---

## Research Questions

This project investigates three main questions:

1. Can explicit latent-space alignment improve generalization from historical to future climates?
2. How are **latent alignment, physical/climate invariance, and OOD predictive performance** related?
3. Do pretrained climate models such as **ClimaX** develop climate-invariant representations without an explicit alignment objective?

---

## Key Contributions

- **Designed a normalized Sliced Wasserstein alignment loss** to prevent latent-space shrinkage when aligning CERA representations across climate scenarios.
- Reduced OOD precipitation prediction degradation by **~80% compared with ClimaX** from Historical to SSP5-8.5 (**R² drop: 0.04 vs. 0.21**).
- Built a large-scale representation-analysis pipeline evaluated on up to **2.5M climate samples** across historical and future climate scenarios.
- Analyzed **latent geometry and cross-climate nearest neighbors** to quantify representation alignment across climate distributions.
- Applied **generalized eigenanalysis** to identify latent directions associated with physically invariant climate relationships.
- Found empirical links between representation alignment, physical/climate invariance, and OOD performance, including evidence that these properties can emerge in **ClimaX without explicit alignment**.

---

## Method Overview

The experiments use daily climate fields from **CESM2 / CMIP6**, covering historical and future SSP climate scenarios.

Climate states are represented as multivariate spatial patches containing **16 atmospheric and surface variables**. The main representation-learning model, **CERA**, combines an autoencoder, a downstream prediction objective, and cross-climate latent alignment.

A central contribution of this work is a **normalized Sliced Wasserstein alignment objective**. Standard distribution alignment can encourage undesirable rescaling or shrinkage of the latent space; the normalized formulation was introduced to align representations across climates while preserving meaningful latent geometry.

CERA representations are evaluated against several baselines, including the **ClimaX climate foundation model**, using both downstream prediction performance and representation-level diagnostics.

### Experimental Pipeline

```text
CESM2 / CMIP6
      │
      ▼
Climate data preprocessing
      │
      ▼
Multivariate spatial patches
      │
      ▼
CERA / ClimaX / baseline models
      │
      ▼
Latent representations
      │
      ▼
OOD performance & invariance analyses
      │
      ▼
Aggregated results and scientific figures
```

---

## Selected Results

### OOD Climate Generalization

The normalized alignment objective substantially improves the stability of precipitation prediction under climate distribution shift.

| Model | Historical R² | SSP5-8.5 R² | R² Drop |
|---|---:|---:|---:|
| **CERA + normalized alignment** | **0.91** | **0.87** | **0.04** |
| ClimaX | 0.58 | 0.37 | 0.21 |

This corresponds to an approximately **81% reduction in OOD performance degradation** relative to ClimaX.

![Precipitation prediction across climate scenarios](figures/r2_precipitation_prediction_across_setups.png)

### Representation Alignment & Physical Invariance

Beyond predictive performance, the project investigates whether climate scenarios become aligned in latent space and whether physically corresponding climate states retain common structure across climate distributions.

The analyses combine **latent-space geometry, cross-climate nearest neighbors, climate separability diagnostics, and generalized eigenanalysis** to study the relationship between representation alignment, physical invariance, and OOD generalization.

---

## Data

Experiments use **CESM2 simulations from CMIP6**, including:

- Historical climate simulations
- Future **SSP1-2.6, SSP2-4.5, SSP3-7.0, and SSP5-8.5** scenarios
- **16 climate variables**
- Daily multivariate spatial patches of approximately **1,000 × 1,000 km**
- Tropical domain spanning approximately **30°S–30°N**
- Up to **2.5 million samples** in the final analyses

The underlying CMIP6 datasets and large model outputs are not stored in this repository due to their size. See [`AVAILABLE_DATA.md`](AVAILABLE_DATA.md) for an inventory of the experiments and artifacts generated during the project.

---

## Tech Stack

**Python · PyTorch · xarray · NumPy · scikit-learn · Matplotlib · Jupyter · PBS/HPC**

The project includes large-scale climate-data preprocessing, deep-learning model training, batch execution on HPC infrastructure, representation analysis, model benchmarking, and scientific visualization.

---

## Repository Structure

```text
Climate-Invariant-Representation-Learning/
│
├── jobs/
│   ├── *.py                    # Climate-data pipeline and batch-execution scripts
│   └── *.pbs                   # PBS jobs for GPU/HPC execution
│
├── notebooks/
│   ├── trainings/
│   │   ├── variable/           # CERA, ClimaX, and baseline training for
│   │   │                       # climate-variable prediction
│   │   └── mask/               # Autoencoder training for masked-point
│   │                           # reconstruction experiments
│   │
│   └── analysis/               # OOD performance, latent geometry,
│                               # climate alignment, invariance, and
│                               # hyperparameter analyses
│
├── figures/                    # Selected figures displayed in this README
│
├── report/
│   ├── reportGraph/
│   │   ├── csv/                # Aggregated metrics used for final analyses
│   │   ├── Graph/              # Figures generated for the research report
│   │   └── CMIP_reportGraph.ipynb
│   │                           # Final figure-generation notebook
│   │
│   └── Titouan_Salin_Columbia_Interim_Research_Report.pdf
│                               # Full internship research report
│
├── AVAILABLE_DATA.md           # Inventory of trained models, experiments,
│                               # sample sizes, and available outputs
│
├── .gitignore                  # Excludes large datasets, checkpoints,
│                               # outputs, and environment-specific files
│
└── README.md                   # Project overview and documentation
```

---

## Experiments

The repository contains several generations of experiments, progressively moving from standard autoencoders toward explicit climate-invariant representation learning:

- **exp3** — CNN autoencoders trained with different levels of climate diversity.
- **exp4** — Adds cross-climate alignment using Sliced Wasserstein Distance.
- **exp5 / CERA** — Final architecture combining reconstruction, downstream prediction, and latent alignment.

The final experiments compare CERA with several controls and baselines, including:

- **CERA without alignment**, isolating the effect of the alignment objective
- **ClimaX**, using representations from a pretrained climate foundation model
- **Simple learned baselines**
- **Physical baselines**
- Variants modifying the aligned latent dimensions and seasonal information

Training is evaluated on both **climate-variable prediction** and **masked-point reconstruction** tasks.

---

## Reproducibility & HPC Workflow

Large-scale experiments were run on HPC infrastructure using **PBS batch jobs**.

The `jobs/` directory contains the data-processing pipeline and Python/PBS wrappers used to execute training and analysis notebooks with different experimental configurations. Training outputs include model checkpoints, evaluation metrics, latent representations, and run metadata.

Large CESM2 datasets, checkpoints, and intermediate model outputs are intentionally excluded from GitHub. [`AVAILABLE_DATA.md`](AVAILABLE_DATA.md) documents the experimental configurations and artifacts generated during the research project.

---

## Research Report

A detailed description of the scientific motivation, methodology, experiments, and results is available in the full internship research report:

### [Read the Columbia Research Report](report/Titouan_Salin_Columbia_Interim_Research_Report.pdf)

---

## Research Context

This work was conducted during a **research internship at Columbia University's Gentine Lab** from **April to August 2026**.

The project lies at the intersection of **machine learning, climate science, representation learning, and out-of-distribution generalization**, with the broader goal of understanding how learned climate representations behave under future climate distribution shifts.

A manuscript building on this research is currently **in preparation**.

---

## Acknowledgements

I am grateful to **Sophie Abramian** for her supervision throughout the project and to **Pierre Gentine** and the Gentine Lab at Columbia University for the research environment and scientific guidance.

---

## Author

**Titouan Salin**  
MScAC — Artificial Intelligence, University of Toronto  
Diplôme d'Ingénieur — Artificial Intelligence & Data Science, École Polytechnique
