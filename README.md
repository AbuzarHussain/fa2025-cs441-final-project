# CS 441 Final Project

How Long Will the Paw Stay? - Predicting Dog Length of Stay at the Champaign County Humane Society

We aim to predict the Length-of-Stay (LOS) for dogs entering the Champaign County Humane Society (CCHS). Using historical intake and adoption data, we will build a machine learning model that predicts how many days a dog is likely to stay in the shelter.
The final deliverable will be a simple web-hosted interface, where shelter staff can input basic dog features and receive a LOS prediction. This meets the CS441 requirement of creating a working ML system with a user-facing demo.

# Team Members
1. Mohammad, Abuzar Hussain <ahm7@illinois.edu>
2. Xu, Duorui <duorui2@illinois.edu>
3. Hung, Brian <hhhung2@illinois.edu>
4. Yu, Zhezhao <zhezhao3@illinois.edu>

# Input
Dog profile (breed/breed group, age, sex & spay/neuter, size/weight, color/coat, microchip, basic health flags), Intake info (type/reason, condition, owner‑surrender vs. stray, prior returns), Context (date, day‑of‑week/season) and Free‑text notes will be converted into a few keyword indicators.

# Output
Regression: Predicted LOS in days (intake → adoption)

Classification: Adopt‑within‑K‑days where K ∈ {7, 14, 30}

# Performance Measure

Regression (LOS): MAE, RMSE, R²

Classification (K‑day): Accuracy, ROC‑AUC, plus a brief calibration check (reliability curve)


