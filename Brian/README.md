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
