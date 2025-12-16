# Xu — LOS Prediction Experiments (CS441 Final Project)

This directory contains **Duorui Xu**’s experiments on predicting **dog length-of-stay (LOS)** in animal shelters using tabular features and neural models.

The work focuses on **binary classification of LOS thresholds**, with tree-based models used as baselines and survival-style experiments included as exploratory attempts.

---

## 1. Task Definition

Given intake and outcome records for shelter dogs, we predict whether the **length of stay (LOS)** exceeds predefined thresholds.

### Primary Tasks (Binary Classification)
We formulate LOS prediction as three binary tasks:

- **LOS ≤ 7 days**
- **LOS ≤ 14 days**
- **LOS ≤ 30 days**

Each task is trained and evaluated independently.

### Scope
- Species: **Dogs only**
- Optional filter: **Adoption-only outcomes**
- Datasets merged from multiple years (Dallas + Champaign style intake/outcome records)

---

## 2. Main Model: Binary MLP

### Model Architecture
- Fully-connected **MLP**
- Hidden layers: `256 → 128`
- Activations: ReLU
- Regularization: BatchNorm + Dropout (0.3 / 0.2)
- Output: single logit (binary classification)

### Training Setup
- Optimizer: Adam
- Learning rate: `1e-3`
- Weight decay: `1e-4`
- Epochs: `80`
- Batch size: `512`
- Label smoothing: `0.05`
- Mixed precision (AMP) on CUDA when available

### Evaluation Protocol
- Stratified split: **70% train / 15% validation / 15% test**
- Repeated runs: **10 runs per task**
- Metrics reported as **mean ± standard deviation**

---

## 3. Feature Engineering

### Demographic & Health Features
- Age group → mapped to approximate days (puppy / adult / senior)
- Log-transformed age (`log(1 + age_days)`)
- Intake condition normalized into:
  - healthy / injured / sick / critical / dead / underage / geriatric / other

### Breed Features
- Breed role/use (herding, sporting, working, terrier, toy, etc.)
- Breed size (small / medium / large / giant)
- Temperament heuristic (fierce vs normal)
- Top-20 most frequent breeds (one-hot)
- `Breed_Other` indicator

### Temporal Features
- Intake month, year, weekday
- Weekend flag
- Holiday flag (if available)
- Intake time-of-day bin (night / morning / afternoon / evening)

### Interaction Features
- Breed use × health condition
- Breed size × health condition
- Temperament × health condition
- Weekend × time-of-day

### Encoding
- Numeric features: standardized
- Categorical features: one-hot encoding

---

## 4. Optional Text Representation (BERT)

An optional branch augments tabular features with **BERT embeddings**:

- Model: `bert-base-uncased`
- Input text concatenates:
  - Animal breed
  - Intake condition
  - Intake reason
  - Hold request
- Embedding: `[CLS]` token
- Max length: 64
- Embeddings cached locally to avoid recomputation

When enabled, BERT features are concatenated with tabular features before training.

---

## 5. Feature Selection

If the final feature dimension exceeds **256**, we apply:

- **L1-regularized Logistic Regression (saga solver)**  
- Features ranked by absolute coefficient magnitude  
- Top-256 features retained

This keeps the MLP input size manageable while preserving predictive signal.

---

## 6. Results — Binary LOS Classification

### Test Performance Summary

| LOS Threshold | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|--------------|----------|-----------|--------|----------|---------|
| ≤ 7 days     | 0.885    | 0.891     | 0.965  | 0.927    | 0.878   |
| ≤ 14 days    | 0.766    | 0.751     | 0.911  | 0.823    | 0.882   |
| ≤ 30 days    | 0.716    | 0.535     | 0.451  | 0.489    | 0.926   |

### Observations
- Short- and medium-term LOS predictions are highly accurate.
- Performance drops for **long-stay dogs (>30 days)** due to:
  - Severe class imbalance
  - Weaker signal in intake-time features
- ROC-AUC remains high even when F1 declines, indicating threshold sensitivity.

---

## 7. Baseline Models (Tree Family)

Tree-based models are used as **baselines**, not primary methods:

- Decision Tree (DT)
- Random Forest (RF)
- Extra Trees (ET)
- Gradient Boosting (GB)

### Baseline Evaluation
- Target: discretized LOS buckets
- Metrics:
  - Weighted F1
  - Concordance-style score (proxy for ranking LOS)

Results are exported to:
- `survclass_tree_family_summary.txt`
- `survclass_results_all_configs.csv`

---

## 8. Additional Experiments (Exploratory)

Survival-style and alternative formulations (referred to as *surv / survclass*) are included as **exploratory attempts**.

These experiments:
- Explore alternative objectives
- Are not treated as the main reported results
- Serve as supporting investigation

---

## 9. Key Takeaways

- Rich feature engineering is critical for tabular LOS prediction.
- MLP models outperform tree baselines for binary LOS thresholds.
- Long-stay prediction is fundamentally challenging due to imbalance.
- Threshold tuning is essential for deployment-oriented use cases.

---

## 10. Future Work

- Log-transform LOS or use robust regression losses (MAE / Huber)
- Focal or cost-sensitive loss for long-stay classification
- Time-based splits for stricter generalization testing
- Deeper integration of text embeddings with tabular models

