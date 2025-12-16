# Dog Shelter LOS Prediction (Tree Ensembles & CatBoost)

This folder contains experiments for predicting **Length of Stay (LOS)** of shelter dogs using **tree-based ensembles** and **CatBoost** on tabular features. The work covers both **regression** (continuous LOS) and **binary classification** (adopt-within-K-days).

---

## 1. Problem Definition

### Tasks
1. **Regression:** predict continuous LOS in days.
2. **Classification:** predict whether adoption occurs **within K days**.

Thresholds used for classification:
- **K = 7, 14, 30**

### Data Filters
- Species: **DOG**
- Outcome: **ADOPTION only**
- LOS is computed from intake/outcome timestamps (or dataset-provided custody days when available).

---

## 2. Feature Engineering

A shared feature pipeline is used across models.

### Temporal Features
- Intake month, weekday, quarter
- Day-of-month, week-of-year
- Weekend indicator

### Categorical Features (when available)
- Breed
- Intake type / subtype
- Intake condition
- Chip status
- Animal origin
- Council district
- **Breed grouping**: rare breeds merged into a single bucket (e.g., `RARE_BREED`) to reduce sparsity

### Encodings / Derived Features
- **Frequency encoding** for high-cardinality categories:
  - breed, intake subtype, council district
- Binary indicators:
  - has chip, is stray, is owner surrender
- Interaction features (explicit for Random Forest):
  - intake type × subtype
  - breed × chip status

> CatBoost consumes raw categorical columns directly (after casting to string), while Random Forest requires label encoding.

---

## 3. Models

### Random Forest
- Strong baseline for tabular learning
- Requires explicit categorical encoding and interaction feature construction
- Hyperparameters tuned via randomized search

### CatBoost
- Native categorical feature handling
- Automatic interactions + built-in regularization
- Early stopping supported

---

## 4. Results

### 4.1 Dallas — Regression (Predict LOS)
| Model | Test R² | RMSE (days) | MAE (days) |
|------|--------:|------------:|-----------:|
| Random Forest | **38.81%** | 12.73 | **6.80** |
| CatBoost | 36.56% | **12.70** | 6.82 |

**Generalization (Train–Test R² Gap)**
- Random Forest: **12.65%**
- CatBoost: **1.86%**

Interpretation: Random Forest slightly leads on peak test R², while CatBoost shows stronger overfitting control.

---

### 4.2 Dallas — Classification (Adopt within K days)

**ROC-AUC (%)**
| K (days) | Random Forest | CatBoost |
|---------:|--------------:|---------:|
| 7 | **80.71** | 79.39 |
| 14 | **76.78** | 76.64 |
| 30 | **73.41** | 70.17 |

**Accuracy (%)**
| K (days) | Random Forest | CatBoost |
|---------:|--------------:|---------:|
| 7 | **75.32** | 73.92 |
| 14 | **83.83** | 83.40 |
| 30 | 94.67 | 94.67 |

Interpretation: Both models perform similarly for mid thresholds; performance degrades as K increases and the task becomes more imbalanced/ambiguous.

---

### 4.3 Champaign — Regression (Predict LOS)
| Model | Test R² | RMSE (days) | MAE (days) |
|------|--------:|------------:|-----------:|
| Random Forest | **15.40%** | **36.06** | 23.27 |
| CatBoost | 13.91% | 36.38 | **23.15** |

Interpretation: performance is substantially lower than Dallas, consistent with smaller sample size and reduced feature availability. CatBoost again shows slightly better stability in error behavior (MAE) while Random Forest achieves slightly higher R².

---

## 5. Visualizations (Static)

Markdown itself does not “compute” plots, but it can **display** plots rendered offline and committed as images.

### Champaign — Model Comparison
![Champaign Model Comparison](https://raw.githubusercontent.com/AbuzarHussain/fa2025-cs441-final-project/main/Yu/champaign_model_comparison.png)

### Dallas — Model Comparison
![Dallas Model Comparison](https://raw.githubusercontent.com/AbuzarHussain/fa2025-cs441-final-project/main/Yu/model_comparison.png)

If images do not render:
- confirm the file exists in the repo at the same path
- ensure branch is `main`
- use `raw.githubusercontent.com/.../main/...png` form (not the GitHub “blob” URL)

---

## 6. Key Takeaways

- Tree ensembles are strong tabular baselines for LOS prediction.
- CatBoost reduces preprocessing burden and improves overfitting control.
- Dataset scale and feature richness dominate performance:
  - Dallas (large + richer features) >> Champaign (smaller + fewer fields).

---

## 7. Limitations & Next Steps

- LOS has a heavy tail; regression metrics can be dominated by long-stay cases.
- Consider:
  - log-transform LOS for regression
  - survival-style objectives (time-to-event) for better tail modeling
  - time-based splits for stricter evaluation
  - calibration and threshold tuning for deployment-oriented classification

