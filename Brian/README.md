# Brian Hung — Tabular MLP Models for LOS Prediction (Champaign Shelter)


## Overview


This component focuses on **tabular deep learning** for predicting dog **Length of Stay (LOS)** at the Champaign County Animal Shelter.
The work explores **both regression and classification formulations**, supported by extensive feature engineering and a carefully regularized **MLP (Multi-Layer Perceptron)** pipeline.


Two complementary tasks are considered:


- **Regression**: predict continuous LOS (days)
- **Classification**: predict whether LOS exceeds a threshold (**7 / 14 / 30 days**)


---


## Modeling Strategy


```mermaid
mindmap
root((MLP-based LOS Prediction))
Data
Champaign Shelter (Dogs only)
Tasks
Regression
Predict LOS (days)
Classification
LOS > 7 days
LOS > 14 days
LOS > 30 days
Feature Engineering
Demographics
Age (days, log-age)
Age group
Breed size
Mixed breed flag
Has name
Temporal
Month / Weekday
Season / Quarter
Holiday & Weekend flags
Cyclical encoding (sin/cos)
Encoding
One-hot (categorical)
Frequency encoding
Target encoding
Interactions
Age × Intake type
Breed size × Intake type
Model
MLP
BatchNorm
Dropout
ReLU
Training
Adam / AdamW
LR scheduling
Early stopping
Gradient clipping
Evaluation
Regression
MAE / RMSE / R²
Classification
Accuracy / F1
ROC-AUC
PR-AUC
Confusion Matrix
```


---


## Feature Engineering


A major emphasis of this work is **high-quality feature engineering** tailored for tabular data:


- **Age features**: age in days, log-age, age group (puppy / adult / senior)
- **Breed features**: simplified breed size, mixed-breed indicator, top-breed one-hot encoding
- **Temporal features**: intake month, weekday, season, quarter, holidays, weekends
- **Cyclical encoding**: sine/cosine transforms for month and weekday
- **Encoding strategies**:
- One-hot encoding (fit on training data only)
- Frequency encoding (train-only statistics)
- Target encoding with smoothing to reduce overfitting
- **Interaction features**: age × intake type, breed size × intake type


All encodings are computed **after train/validation/test split** to avoid data leakage.

---

## Regression Results — Predicting Continuous LOS

### Training Dynamics
![Regression Training Curves](https://raw.githubusercontent.com/AbuzarHussain/fa2025-cs441-final-project/main/Brian/Champaign/champaign_mlp_regression_evaluation.png)

- Training loss decreases steadily.
- Validation loss plateaus early, indicating overfitting.

### Predicted vs Actual LOS
![Predicted vs Actual LOS](https://raw.githubusercontent.com/AbuzarHussain/fa2025-cs441-final-project/main/Brian/Dallas/mlp_predicted_vs_actual.png)

- The model systematically underpredicts long-stay dogs.
- This reflects the heavy-tailed distribution of LOS values.

---

## Classification Results — LOS Threshold Prediction

### Confusion Matrices
![Classification Confusion Matrices](https://raw.githubusercontent.com/AbuzarHussain/fa2025-cs441-final-project/main/Brian/Dallas/mlp_classification_confusion_matrices.png)

---

### ROC Curves
![ROC Curves](https://raw.githubusercontent.com/AbuzarHussain/fa2025-cs441-final-project/main/Brian/Dallas/mlp_classification_roc_curves.png)

---

### Precision–Recall Curves
![PR Curves](https://raw.githubusercontent.com/AbuzarHussain/fa2025-cs441-final-project/main/Brian/Dallas/mlp_classification_pr_curves.png)

---

### Training Loss Curves
![Training Curves](https://raw.githubusercontent.com/AbuzarHussain/fa2025-cs441-final-project/main/Brian/Dallas/mlp_classification_training_curves.png)

---

### Regression Loss Curve
![Regression Loss](https://raw.githubusercontent.com/AbuzarHussain/fa2025-cs441-final-project/main/Brian/Dallas/mlp_loss_curve.png)

