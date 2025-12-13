# CS 441 Final Project

How Long Will the Paw Stay? - Predicting Dog Length of Stay at the Champaign County Humane Society

We aim to predict the Length-of-Stay (LOS) for dogs entering the Champaign County Humane Society (CCHS). Using historical intake and adoption data, we will build a machine learning model that predicts how many days a dog is likely to stay in the shelter.
The final deliverable will be a simple web-hosted interface, where shelter staff can input basic dog features and receive a LOS prediction. This meets the CS441 requirement of creating a working ML system with a user-facing demo.

# Team Members
1. Mohammad, Abuzar Hussain <ahm7@illinois.edu>
2. Xu, Duorui <duorui2@illinois.edu>
3. Hung, Brian <hhhung2@illinois.edu>
4. Yu, Zhezhao <zhezhao3@illinois.edu>

# Data Sources

This repository contains two datasets related to dog intake and outcome records from animal shelters. The datasets were obtained from different sources and have different access conditions, as described below.

1. Dallas Animal Services Data (Public Open Data)

File: Intake and Outcome 11-1-23 through 10-31-24.xlsx

Geographic coverage: Dallas, Texas

Source: City of Dallas Open Data Portal (Dallas Animal Services)

Access method: Publicly available dataset downloaded directly from the official open data website and exported as an Excel file.

Data portal:

Dallas Open Data – Dallas Animal Shelter Data (Fiscal Year 2023–2026)

Platform: Socrata Open Data

Dataset ID: uyte-zi7f

https://www.dallasopendata.com/

The file name reflects the specific date range selected at the time of download (November 1, 2023 through October 31, 2024). The data represent a snapshot of the larger continuously updated dataset maintained by Dallas Animal Services.

2. Champaign Humane Society Data (By Request https://www.cuhumane.org/)

File: dogs_intake_outcome_2021_2025.xlsx

Geographic coverage: Champaign, Illinois

Source: Champaign Humane Society

Access method: Obtained through direct communication with the organization.

Data acquisition: The dataset was provided after coordination between Duorui Xu and Tief (supervisor at Champaign Humane Society).

This dataset is not publicly downloadable and was shared for research and analysis purposes. It represents an internal export covering dog intake and outcome records from 2021 to 2025.

# Input
Dog profile (breed/breed group, age, sex & spay/neuter, size/weight, color/coat, microchip, basic health flags), Intake info (type/reason, condition, owner‑surrender vs. stray, prior returns), Context (date, day‑of‑week/season) and Free‑text notes will be converted into a few keyword indicators.

# Output
Regression: Predicted LOS in days (intake → adoption)

Classification: Adopt‑within‑K‑days where K ∈ {7, 14, 30}

# Performance Measure

Regression (LOS): MAE, RMSE, R²

Classification (K‑day): Accuracy, ROC‑AUC, plus a brief calibration check (reliability curve)


