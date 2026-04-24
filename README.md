# DySCo fMRI Analysis Pipeline

**Dynamic Symmetric Connectivity (DySCo) applied to generalised epilepsy and healthy controls**

---

## Abstract

Epilepsy is increasingly understood as a disorder of large-scale brain network dynamics rather than purely localised electrical dysfunction. This study applies the Dynamic Symmetric Connectivity (DySCo) framework to investigate dynamic functional connectivity (dFC) in paediatric (n=5) and adult (n=10) generalised epilepsy cohorts, alongside healthy controls (n=19), using functional magnetic resonance imaging (fMRI) data acquired during cartoon-viewing and resting-state conditions. Four complementary DySCo metrics were examined: von Neumann entropy, reconfiguration speed, connectivity norm, and metastability.

Group-level statistical comparisons revealed no significant differences between cartoon and resting-state conditions across any cohort. However, directional trends were observed in the paediatric epilepsy group, with four of five participants showing higher entropy during cartoon viewing and higher connectivity norm and reconfiguration speed during rest. In contrast, adult epilepsy participants showed minimal effect sizes and heterogeneous responses, while healthy controls exhibited near-zero effect sizes with balanced directional splits, indicating no systematic condition difference.

Within-cartoon analyses revealed that the embedded waiting periods within the cartoon paradigm systematically modulated connectivity dynamics. Both epilepsy cohorts demonstrated a progressive decrease in entropy during waiting periods, while reconfiguration speed and connectivity norm increased. This pattern was absent in healthy controls, who showed stable entropy across the same interval. This dissociation suggests that onset-aligned entropy dynamics during stimulus withdrawal may reflect a network process specifically altered in generalised epilepsy.

Resting-state entropy increased across cohorts from paediatric epilepsy (~2.10–2.20) to adult epilepsy (~2.23–2.26) and healthy controls (~2.255–2.260), consistent with neurodevelopmental maturation of network complexity. Functional Connectivity Dynamics (FCD) analysis further identified consistent paradigm-driven structures across all groups, alongside reduced recurrence of connectivity states in epilepsy relative to controls.

Overall, while primary condition contrasts were not statistically significant, the results demonstrate the sensitivity of DySCo metrics — particularly von Neumann entropy — to temporal structure within paradigms. The entropy response to stimulus removal emerges as a promising candidate marker of altered network dynamics in epilepsy, though validation in larger, age-matched cohorts is required for clinical translation.

---

## Repository Structure

```
dissertation_code/
├── core_functions/               # DySCo algorithm implementations
│   ├── compute_eigenvectors_sliding_cov.py   # Sliding-window covariance eigenvectors
│   ├── dysco_distance.py                     # DySCo distance metric
│   ├── dysco_entropy.py                      # Von Neumann entropy
│   └── dysco_norm.py                         # Connectivity norm (L2)
│
├── pipeline/                     # Main pipeline and cohort run scripts
│   ├── dysco_nifti_pipeline.py               # Core pipeline: NIfTI → .npy → figures
│   ├── run_paediatric_all_patients.py        # Paediatric epilepsy cohort (n=5)
│   ├── run_adult_all_patients.py             # Adult epilepsy cohort (n=10)
│   └── run_hc_all_patients.py               # Healthy controls (n=19)
│
├── figures/
│   ├── fcd/                      # Functional Connectivity Dynamics matrices
│   │   ├── generate_fcd_matrices.py          # Paediatric FCD (4-panel)
│   │   ├── generate_adult_fcd_matrices.py    # Adult FCD (4-panel)
│   │   └── generate_hc_fcd_matrices.py       # HC FCD (4-panel)
│   │
│   ├── timecourses/              # Within-cartoon DySCo timecourses
│   │   ├── generate_within_cartoon_all_patients.py   # Paediatric
│   │   ├── generate_adult_within_cartoon.py          # Adult
│   │   └── generate_hc_within_cartoon.py             # Healthy controls
│   │
│   ├── boxplots/                 # Three-condition boxplots (Video / Wait / Rest)
│   │   ├── generate_three_condition_boxplots.py      # Paediatric
│   │   ├── generate_adult_group_boxplot.py           # Adult
│   │   └── generate_hc_group_boxplot.py              # Healthy controls
│   │
│   └── cross_group/              # Cross-cohort comparisons
│       └── generate_cross_group_comparison.py
│
└── tables/                       # Statistical tables (Wilcoxon, subject means)
    └── make_table.py
```

---

## Methods Overview

### DySCo Framework

DySCo characterises dynamic functional connectivity by computing a sliding-window covariance matrix over the fMRI BOLD timeseries and extracting leading eigenvectors at each timepoint. Four metrics are derived:

| Metric | Description |
|---|---|
| **Von Neumann Entropy** | Spectral entropy of the connectivity state; higher = more distributed/uncertain |
| **Reconfiguration Speed** | Rate of change between consecutive connectivity states (with lag) |
| **Connectivity Norm (L2)** | Overall magnitude of the connectivity state |
| **FCD Matrix** | T×T matrix of DySCo distances between all pairs of timepoints |

### Parameters

| Parameter | Value |
|---|---|
| TR | 2.16 s |
| Half-window size | 10 volumes (full window = 21 TRs) |
| Number of eigenvectors | 10 |
| Speed lag | 20 volumes |
| Edge trim | 10 windows per run boundary |

### Paradigm (Shamshiri et al., 2016)

Cartoon runs follow a fixed block structure:

```
Video 1:     volumes   0–111
Wait 1:      volumes 111–150
Video 2:     volumes 150–261
Wait 2:      volumes 261–296 (paediatric) / 261–300 (adult/HC)
```

---

## Cohorts

| Cohort | n | Condition | Sessions |
|---|---|---|---|
| Paediatric generalised epilepsy | 5 | Cartoon + Rest | 1 |
| Adult generalised epilepsy | 10 | Cartoon + Rest | ses-01 |
| Healthy controls | 19 | Cartoon + Rest | ses-01 |

---

## Dependencies

```
numpy
scipy
matplotlib
nibabel
tqdm
pandas
```

Install with:
```bash
pip install numpy scipy matplotlib nibabel tqdm pandas
```

---

## Usage

1. Set paths in the relevant run script (`run_paediatric_all_patients.py`, etc.)
2. Set `SKIP_PROCESSING = False` for first run (converts NIfTI → `.npy`)
3. Run the pipeline:

```bash
python dissertation_code/pipeline/run_paediatric_all_patients.py
python dissertation_code/pipeline/run_adult_all_patients.py
python dissertation_code/pipeline/run_hc_all_patients.py
```

4. Generate figures:

```bash
# FCD matrices
python dissertation_code/figures/fcd/generate_fcd_matrices.py

# Within-cartoon timecourses
python dissertation_code/figures/timecourses/generate_within_cartoon_all_patients.py

# Condition boxplots
python dissertation_code/figures/boxplots/generate_three_condition_boxplots.py

# Cross-group comparison
python dissertation_code/figures/cross_group/generate_cross_group_comparison.py
```


