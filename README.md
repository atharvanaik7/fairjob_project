# FairJob Exposure Fairness

**From Prediction to Control: Fairness in Job Ad Exposure Using the FairJob Dataset**  
Atharva Naik · Shreyas Pasumarthi  
DSCI 599 · University of Southern California · 2025  
https://github.com/atharvanaik7/fairjob_project

---

## Overview

This project studies fairness in job ad exposure using the real-world [FairJob (Criteo)](https://huggingface.co/datasets/criteo/FairJob) dataset — 1,072,226 impressions across 13,568 ranking sessions with a binary protected attribute and binary seniority labels. We train a LightGBM model to predict click probability, evaluate exposure fairness using position-sensitive metrics (DCG@10, average rank), and test two post-processing interventions: score-based re-ranking and FA\*IR constrained re-ranking.

**Key findings:**
- Group 0 is consistently disadvantaged: DCG gap of −0.282 and exposure gap of −0.061 despite near-identical click rates, confirming a ranking-stage rather than prediction-stage phenomenon
- Score-based re-ranking partially corrects at λ=0.001 (gap narrows to −0.138) but overcorrects at λ=0.005 (gap flips to +0.084) with no λ producing a proportional correction
- FA\*IR is a complete null result across all p values — the dataset's balanced 50/50 group composition already satisfies the representation constraint without reordering
- The gap worsens monotonically across five temporal windows, from −0.174 in W1 to −0.388 in W5, consistent with a click feedback loop compounding disadvantage over time

---

## Requirements

- Python 3.11+
- See `requirements.txt`

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
2. Train LightGBM (AUC = 0.853) and logistic regression click predictors
3. Compute baseline fairness metrics across all 13,568 sessions
4. Run score boost sweep (λ = 0, 0.001, 0.005, 0.01, 0.05, 0.1)
5. Run FA\*IR at p = 0.5, 0.7, 0.9
6. Train and evaluate position-debiased model using `displayrandom` impressions
7. Run temporal analysis across 5 session windows
8. Print a full results table

Expected runtime: ~10–20 minutes on a modern CPU.

---

## Project Structure

```
.
├── fairness_analysis.py   # Full pipeline: data, model, metrics, interventions
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
| **Avg. Senior Item Rank** | Mean rank position of senior items per group (lower = better) |

---

## Results

| Method | Exp. Gap | DCG Gap | DCG G0 | DCG G1 |
|---|---|---|---|---|
| Baseline | −0.061 | −0.282 | 1.347 | 1.629 |
| Score boost λ=0.001 | −0.027 | −0.138 | 1.488 | 1.625 |
| Score boost λ=0.005 | +0.021 | +0.084 | 1.707 | 1.623 |
| FA\*IR p=0.5 | −0.061 | −0.281 | 1.347 | 1.629 |
| FA\*IR p=0.7 | −0.061 | −0.281 | 1.347 | 1.628 |
| FA\*IR p=0.9 | −0.061 | −0.281 | 1.348 | 1.628 |
| Position-Debiased | −0.060 | −0.276 | 1.372 | 1.648 |

**Temporal analysis:**

| Window | Exp. Gap | DCG Gap | DCG G0 | DCG G1 |
|---|---|---|---|---|
| W1 (0–2712) | −0.039 | −0.174 | 1.421 | 1.596 |
| W2 (2713–5425) | −0.056 | −0.252 | 1.356 | 1.608 |
| W3 (5426–8138) | −0.055 | −0.258 | 1.355 | 1.613 |
| W4 (8139–10851) | −0.072 | −0.338 | 1.307 | 1.645 |
| W5 (10852–13567) | −0.084 | −0.388 | 1.295 | 1.683 |

---

## Dataset

FairJob is loaded directly from HuggingFace:

```python
from datasets import load_dataset
ds = load_dataset("criteo/FairJob", split="train")
```

The protected attribute identity (which demographic Group 0 and Group 1 represent) is not disclosed by Criteo.

---

## Citation

```
Naik, A. and Pasumarthi, S. (2025). From Prediction to Control: Fairness in
Job Ad Exposure Using the FairJob Dataset. DSCI 599, University of Southern
California.
```
