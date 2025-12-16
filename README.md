# CS 441 Final Project  
## How Long Will the Paw Stay? — Predicting Dog Length of Stay at the Champaign County Humane Society

We aim to predict the **Length-of-Stay (LOS)** for dogs entering the **Champaign County Humane Society (CCHS)**. Using historical intake and adoption data, we build machine learning models that predict how many days a dog is likely to stay in the shelter.

The final deliverable is a **simple web-hosted interface** where shelter staff can input basic dog features and receive a LOS prediction. This meets the CS441 requirement of creating a working ML system with a user-facing demo.

---

## Team Members
- Mohammad, Abuzar Hussain — ahm7@illinois.edu  
- Duorui Xu — duorui2@illinois.edu  
- Brian Hung — hhhung2@illinois.edu  
- Zhezhao Yu — zhezhao3@illinois.edu  

---

## Data Sources

This repository contains two datasets related to dog intake and outcome records from animal shelters. The datasets were obtained from different sources and have different access conditions.

### 1) Dallas Animal Services Data (Public Open Data)
- **File:** `Intake and Outcome 11-1-23 through 10-31-24.xlsx`  
- **Geographic coverage:** Dallas, Texas  
- **Source:** City of Dallas Open Data Portal (Dallas Animal Services)  
- **Access method:** Publicly available dataset downloaded from the official open data website and exported as an Excel file  
- **Platform:** Socrata Open Data  
- **Dataset ID:** `uyte-zi7f`  
- **Portal:** `https://www.dallasopendata.com/`

> The file name reflects the specific date range selected at the time of download (Nov 1, 2023 through Oct 31, 2024). The data represent a snapshot of a continuously updated dataset maintained by Dallas Animal Services.

### 2) Champaign Humane Society Data (By Request)
- **Organization:** Champaign Humane Society — `https://www.cuhumane.org/`  
- **File:** `dogs_intake_outcome_2021_2025.xlsx`  
- **Geographic coverage:** Champaign, Illinois  
- **Source:** Champaign Humane Society  
- **Access method:** Obtained through direct communication with the organization  
- **Data acquisition:** Provided after coordination between Duorui Xu and Mary Tiefenbrunn (“Tief”), Executive Director at Champaign Humane Society

> This dataset is not publicly downloadable and was shared for research and analysis purposes. It is an internal export covering dog intake and outcome records from 2021 to 2025.

---

## Input

Dog profile features and intake context are used to make predictions:

- **Dog profile:** breed / breed group, age, sex & spay/neuter, size/weight, color/coat, microchip status, basic health flags  
- **Intake info:** type/reason, condition, owner-surrender vs. stray, prior returns  
- **Context:** date, day-of-week / season  
- **Free-text notes:** converted into keyword indicators or text embeddings (final model)

---

## Output

### Regression
- **Predicted LOS** in days (intake → adoption)

### Classification
- **Adopt-within-K-days**, where **K ∈ {7, 14, 30}**

---

## Performance Measure

### Regression (LOS)
- MAE, RMSE, R²

### Classification (K-day)
- Accuracy, ROC-AUC  
- Calibration check (reliability curve)

---

## Modeling Approach and System Design

Our modeling pipeline progressively increases model capacity and information usage, starting from simpler baselines and ending with a deployable final model. Each stage is evaluated independently and motivates the next design choice.

### Overall Roadmap (Arrow View)

Raw Shelter Records  
→ Data Cleaning & Filtering  
→ Feature Engineering  
→ Tabular Baselines (MLP, Tree Models)  
→ Text-Enhanced Model (BERT)  
→ Final Model Selection  
→ Web Demo Deployment

---

### Step 1 — Data Preparation

Raw intake/outcome records  
→ filter to DOG entries  
→ compute LOS in days (Outcome_Date − Intake_Date)  
→ remove invalid / negative durations  
→ optionally restrict to adoption outcomes (intake → adoption)

---

### Step 2 — Feature Engineering (Shared Backbone)

Engineered tabular features used across baselines include:
- **Temporal:** month, weekday, quarter, weekend flag, (optional) holiday indicator  
- **Categorical:** breed / breed group, intake type, intake condition, etc.  
- **High-cardinality handling:** frequency encoding / grouping rare categories  
- **Interactions:** simple combinations (e.g., intake type × condition)

These features support both:
- Regression LOS prediction  
- Classification adopt-within-K prediction (K = 7, 14, 30)

---

### Step 3 — Baseline 1: MLP (Deep Tabular Baseline)

Engineered tabular features  
→ MLP (BatchNorm + ReLU + Dropout)  
→ early stopping / learning-curve monitoring  
→ evaluate regression + K-day classification

**Purpose:** establish a strong neural baseline on structured features and diagnose where tabular signals are insufficient (especially for long-stay cases).

> Report the MLP metrics here (Accuracy / ROC-AUC for K=7,14,30; MAE/RMSE/R² for regression) and link to result plots.

---

### Step 4 — Baseline 2: Tree-Based Models (Classical Tabular Baseline)

Same engineered tabular features  
→ tree family models (e.g., Decision Tree / Random Forest / Extra Trees / Gradient Boosting; and/or CatBoost)  
→ evaluate regression + K-day classification  
→ analyze feature importance for interpretability

**Purpose:** strong tabular baselines that are robust and interpretable, providing a reference point for any gain from text modeling.

> Report tree-model metrics here and summarize the best-performing configuration.

---

### Step 5 — Final Model: Tabular + Text with BERT

Engineered tabular features  
+ free-text intake notes / descriptions  
→ BERT embeddings  
→ fusion with tabular features  
→ prediction head for regression + classification

**Why text is needed:** long-stay outcomes often have weak signal in structured fields alone; unstructured notes can contain health/behavior/history cues that affect adoption timing.

**Final model definition:**

> **Final Model = (Tabular engineered features) + (BERT text embeddings) → unified predictor**

> Report final-model metrics here and clearly state whether the final submission uses:
> - regression only, classification only, or both,
> - and what thresholding strategy (if any) is used for deployment.

---

## Demo System (Planned Deliverable)

The final system will expose a web UI where shelter staff can:
1) input dog intake features (and optional notes),  
2) receive predicted LOS (days) and/or adopt-within-K probabilities.

---

## Repository Notes
- **Languages:** primarily Jupyter Notebook and Python  
- **Contributors:** 4

