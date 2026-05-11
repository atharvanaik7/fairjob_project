# FairJob Exposure Fairness

**From Prediction to Control: Fairness in Job Ad Exposure Using the FairJob Dataset**  
Atharva Naik · Shreyas Pasumarthi  
DSCI 599 · University of Southern California · 2025

---

## Overview

This project studies fairness in job ad exposure using the real-world [FairJob (Criteo)](https://huggingface.co/datasets/criteo/FairJob) dataset — 1,072,226 impressions with a binary protected attribute and binary seniority labels. We train a LightGBM model to predict click probability, evaluate exposure fairness across demographic groups using position-sensitive metrics (DCG@10, average rank), and test two post-processing interventions: score-based re-ranking and FA\*IR constrained re-ranking.

**Key finding:** Group 0 senior jobs rank 77 positions lower on average than Group 1 despite identical click rates, confirming a ranking-stage bias invisible to prediction metrics like AUC. Score-based re-ranking overcorrects immediately. FA\*IR at p=0.9 reduces the DCG gap from −0.145 to −0.127 but cannot close a 77-position average-rank gap with a top-10 constraint.

---

## Requirements

- Python 3.11+
- See `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Quickstart

```bash
python code/fairness_analysis.py
```

This will:
1. Download FairJob from HuggingFace (~300MB, cached after first run)
2. Train LightGBM and logistic regression click predictors
3. Compute baseline fairness metrics across all 13,568 sessions
4. Run score boost sweep (λ = 0, 0.001, 0.005, 0.01, 0.05, 0.1)
5. Run FA\*IR at p = 0.5, 0.7, 0.9
6. Train and evaluate position-debiased model (if `displayrandom` flag available)
7. Run temporal analysis across 5 session windows
8. Print a full results table

Expected runtime: ~10–20 minutes on a modern CPU (most time is dataset loading and model scoring across 1M rows).

---

## Project Structure

```
.
├── code/
│   └── fairness_analysis.py   # Main pipeline: data, model, metrics, interventions
├── requirements.txt
└── README.md
```

---

## Metrics

| Metric | Description |
|---|---|
| **Senior Exposure Rate** | Fraction of top-10 items that are senior-level, per group |
| **DCG@10** | Position-weighted senior job exposure (logarithmic discount by rank) |
| **NDCG@10** | DCG normalized by ideal DCG for the session |
| **Avg. Senior Item Rank** | Mean rank position of senior items, per group (lower = better) |

---

## Interventions

**Score boost:** Adds a scalar λ to predicted scores of Group 0 senior items before ranking. Even λ=0.001 immediately flips the gap direction due to score clustering near the ranking threshold.

**FA\*IR:** Constrained re-ranking that enforces a minimum proportion p of Group 0 items in each top-10 list. Provides statistical representation guarantees per session. Limited by the top-k scope relative to the full session size.

---

## Dataset

FairJob is loaded directly from HuggingFace:

```python
from datasets import load_dataset
ds = load_dataset("criteo/FairJob", split="train")
```

The protected attribute identity (which demographic Group 0 and Group 1 represent) is not disclosed by Criteo.

---

## Results Summary

| Method | Exp. Gap | DCG Gap | DCG G0 | DCG G1 |
|---|---|---|---|---|
| Baseline | −0.036 | −0.145 | 2.932 | 3.077 |
| Score boost λ=0.001 | +0.039 | +0.146 | 3.475 | 3.066 |
| FA\*IR p=0.9 | −0.033 | −0.127 | 2.942 | 3.069 |
| Position-debiased | −0.036 | −0.165 | 2.814 | 2.978 |

---

## Citation

If you use this code, please cite:

```
Naik, A. and Pasumarthi, S. (2025). From Prediction to Control: Fairness in
Job Ad Exposure Using the FairJob Dataset. DSCI 599, University of Southern
California.
```
