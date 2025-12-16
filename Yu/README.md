# Shelter Length-of-Stay (LOS) Prediction — Dallas & Champaign (Tabular ML)

This folder contains code + results for predicting **dog shelter length-of-stay (LOS)** using **tabular features** on two datasets:

- **Dallas** (large-scale; richer categorical fields)
- **Champaign** (smaller; fewer available features)

Two modeling formulations are covered:

1. **Regression**: predict continuous LOS (days)
2. **Binary Classification**: predict whether adoption occurs **within K days** (K ∈ {7, 14, 30})

---

## Contents

### Dallas
- `Dallas/predict_shelter_stay_catboost.py` — LOS regression (CatBoost baseline)
- `Dallas/predict_adoption_classification.py` — classification for K={7,14,30}
- `Dallas/visualize_model_comparison.py` — produces comparison plots
- `Dallas/model_comparison.png` — exported visualization

### Champaign
- `Champaign/predict_champaign_shelter_stay.py` — Champaign regression + (optional) classification
- `Champaign/visualize_champaign_comparison.py` — produces comparison plots
- `Champaign/champaign_model_comparison.png` — exported visualization

---

## Visual Results (Rendered in GitHub)

### Dallas: Random Forest vs CatBoost
![Dallas Model Comparison](https://raw.githubusercontent.com/AbuzarHussain/fa2025-cs441-final-project/main/Yu/Dallas/model_comparison.png)

### Champaign: Random Forest vs CatBoost
![Champaign Model Comparison](https://raw.githubusercontent.com/AbuzarHussain/fa2025-cs441-final-project/main/Yu/Champaign/champaign_model_comparison.png)

> Notes on GitHub rendering:
> - Use `raw.githubusercontent.com/.../main/...png` for images (works reliably in README).
> - The `::contentReference[oaicite:0]{...}` text is **NOT** GitHub Markdown and should **not** be placed in README.

---

## Problem Setup

### Target Definitions
- **LOS (days)** = (Outcome_Date − Intake_Date) in days  
- **Adoption-only subset (optional)**: many scripts filter to `Outcome_Type == "ADOPTION"` to model adoption LOS consistently.

### Tasks
- **Regression**: predict LOS as a real value (days).
- **Classification**: `y = 1` if LOS ≤ K days, else `0` (for K = 7, 14, 30).

---

## Feature Engineering Summary (Tabular)

Common engineered signals used across scripts:

**Temporal**
- Intake month / weekday / quarter
- Weekend flag
- Day of month, week of year

**Categorical (high-cardinality handling)**
- Raw category columns (e.g., breed, intake type, condition…)
- Rare-category grouping (e.g., rare breeds → `RARE_BREED`)
- Frequency encoding (counts of category appearances)

**Binary flags**
- “Has chip”
- “Stray”
- “Owner surrender”

**Interactions (in some scripts)**
- Intake type × subtype
- Breed × chip status
- Intake type × condition

---

## Results Summary

### Dallas — Regression (LOS)
Comparison of Random Forest vs CatBoost (tabular regression):

| Model | Test R² | RMSE (days) | MAE (days) | Train R² | Train–Test Gap |
|------|---------:|------------:|-----------:|---------:|---------------:|
| Random Forest | 0.3881 (38.81%) | 12.73 | 6.80 | 0.5146 (51.46%) | 12.65% |
| CatBoost | 0.3656 (36.56%) | 12.70 | 6.82 | 0.3842 (38.42%) | 1.86% |

**Interpretation**
- Random Forest achieves **slightly higher test R²**.
- CatBoost shows **much better overfitting control** (smaller train–test gap).

---

### Dallas — Classification (Adopt within K days)
Metrics reported for K = 7 / 14 / 30:

#### ROC-AUC (%)
| K | Random Forest | CatBoost |
|---:|-------------:|---------:|
| 7  | 80.71 | 79.39 |
| 14 | 76.78 | 76.64 |
| 30 | 73.41 | 70.17 |

#### Accuracy (%)
| K | Random Forest | CatBoost |
|---:|-------------:|---------:|
| 7  | 75.32 | 73.92 |
| 14 | 83.83 | 83.40 |
| 30 | 94.67 | 94.67 |

**Interpretation**
- Performance is strongest for **K=30** in terms of accuracy (partly reflecting class distribution).
- ROC-AUC decreases as K grows to 30 for CatBoost in the recorded run.
- For deployment, **threshold tuning** (not always 0.5) can be important when positive/negative costs differ.

---

### Champaign — Regression (LOS)
Champaign uses **fewer available categorical fields** and has a smaller dataset.

| Model | Test R² | RMSE (days) | MAE (days) | Train R² | Train–Test Gap |
|------|---------:|------------:|-----------:|---------:|---------------:|
| Random Forest | 0.1540 (15.40%) | 36.06 | 23.27 | 0.3909 (39.09%) | 23.69% |
| CatBoost | 0.1391 (13.91%) | 36.38 | 23.15 | 0.2638 (26.38%) | 12.47% |

**Interpretation**
- Both models are **much worse than Dallas** on R² (≈14–15% vs ≈36–39%).
- Likely drivers:
  - smaller sample size (≈1.8k dogs vs tens of thousands),
  - fewer strong predictive features,
  - domain differences between cities.

---

## How to Run

> The scripts assume you have the required dataset files locally (CSV/XLSX). Update the file paths at the top of each script.

### Dallas
```bash
cd Yu/Dallas
python predict_shelter_stay_catboost.py
python predict_adoption_classification.py
python visualize_model_comparison.py
