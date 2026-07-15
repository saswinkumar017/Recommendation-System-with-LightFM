# Recommendation System with LightFM

**A Hybrid Recommendation System using Matrix Factorization**  
*CodTech IT Solutions — Machine Learning Internship (Intern ID: CITS5452)*

---

## Overview

A production-style recommendation engine built from scratch in Python that learns user preferences from implicit feedback and generates personalized top-N recommendations. Implements the LightFM algorithm — a hybrid matrix factorization model combining collaborative filtering with content metadata.

---

## Features

- **Matrix Factorization** — learns latent embeddings for users and items
- **3 Loss Functions** — WARP, BPR, Logistic (configurable)
- **Evaluation Suite** — Precision@K, Recall@K, AUC, Ranking Loss
- **Loss Comparison** — benchmarks all three loss functions side-by-side
- **Epoch Analysis** — tracks metrics across training iterations
- **Visualization** — rating distribution, sparsity, embeddings, training curves
- **Model Persistence** — save/load via pickle

---

## Project Structure

```
recommendation-system-with-lightfm/
│
├── dataset/
│   └── ml-100k/           # MovieLens 100K (u.data, u.item, u.user)
│
├── outputs/
│   ├── graphs/            # Generated visualizations (PNG)
│   ├── recommendations/   # Top-N recommendations (CSV)
│   └── trained_model/     # Saved model + config (pickle)
│
├── src/
│   ├── data_loader.py     # Download & load MovieLens
│   ├── preprocessing.py   # Clean, map IDs, build sparse matrix
│   ├── train.py           # LightFM algorithm (WARP/BPR/Logistic)
│   ├── recommend.py       # Generate top-N recommendations
│   ├── evaluate.py        # Precision@K, Recall@K, AUC
│   ├── visualization.py   # Matplotlib graphs
│   └── utils.py           # Config & helpers
│
├── main.py                # Entry point — runs the full pipeline
├── requirements.txt
├── README.md
└── DOCUMENTATION.md        # Complete educational reference
```

---

## Dataset

**MovieLens 100K** — 100,000 ratings from 943 users on 1,682 movies.

| Metric | Value |
|--------|-------|
| Users | 943 |
| Items | 1,682 |
| Interactions | 100,000 |
| Sparsity | 93.70% |
| Rating Scale | 1–5 |

The dataset is downloaded automatically on first run.

---

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

The pipeline runs all steps: download → EDA → clean → train → recommend → evaluate → visualize → save model.

---

## Results (K=10)

| Loss | Precision@K | Recall@K | AUC |
|------|-------------|----------|-----|
| **WARP** | 0.0844 | 0.0786 | 0.2872 |
| **BPR** | 0.1012 | 0.0994 | 0.2852 |
| **Logistic** | 0.0900 | 0.0781 | 0.2963 |

---

## Outputs

| Artifact | Location |
|----------|----------|
| Trained model | `outputs/trained_model/lightfm_model.pkl` |
| Recommendations | `outputs/recommendations/top_recommendations.csv` |
| Evaluation metrics | `outputs/evaluation_metrics.json` |
| Graphs (7) | `outputs/graphs/` |

---

## Configuration

Edit `src/utils.py`:

```python
"model_params": {
    "no_components": 30,     # embedding size
    "learning_rate": 0.05,   # SGD step size
    "max_sampled": 10,       # WARP negative samples
    "random_state": 42,
},
"train_params": {"epochs": 30},
"eval_params": {"k": 10, "test_size": 0.2},
```

---

## Algorithms

| Technique | Description |
|-----------|-------------|
| **Matrix Factorization** | Decomposes user-item matrix into latent user/item embeddings |
| **Negative Sampling** | For each positive interaction, sample unseen items to learn relative preference |
| **WARP Loss** | Weighted Approximate-Rank Pairwise — optimises top-N ranking |
| **BPR Loss** | Bayesian Personalized Ranking — pairwise preference optimisation |
| **Logistic Loss** | Pointwise binary classification |

---

## Internship Details

| Field | Value |
|-------|-------|
| **Organization** | CodTech IT Solutions |
| **Internship** | Machine Learning Internship |
| **Intern ID** | CITS5452 |
| **Language** | Python 3.10 |
| **IDE** | VS Code |

---

*Submitted for CodTech IT Solutions Machine Learning Internship.*
