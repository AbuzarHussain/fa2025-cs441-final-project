"""
CatBoost Model for Predicting Dog Shelter Stay Duration
Best ML Method: CatBoost - Native categorical feature handling, automatic interactions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, classification_report)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================
print("="*70)
print("CATBOOST MODEL: Dog Shelter Stay Duration Prediction")
print("="*70)

# Load the data
print("\n1. Loading data...")
df = pd.read_csv('Dallas_Animal_Shelter_Data_Fiscal_Year_2023_-_2026_20251115.csv')

# Filter for dogs only
print("2. Filtering for dogs only...")
df_dogs = df[df['Animal_Type'] == 'DOG'].copy()

# Calculate stay duration in days
print("3. Calculating stay duration...")
df_dogs['Intake_Date'] = pd.to_datetime(df_dogs['Intake_Date'], errors='coerce')
df_dogs['Outcome_Date'] = pd.to_datetime(df_dogs['Outcome_Date'], errors='coerce')
df_dogs['Stay_Duration_Days'] = (df_dogs['Outcome_Date'] - df_dogs['Intake_Date']).dt.total_seconds() / (24 * 3600)

# Remove invalid records
df_dogs = df_dogs.dropna(subset=['Intake_Date', 'Outcome_Date', 'Stay_Duration_Days'])
df_dogs = df_dogs[df_dogs['Stay_Duration_Days'] >= 0]

print(f"   Total dog records: {len(df_dogs)}")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*70)
print("FEATURE ENGINEERING")
print("="*70)

X = df_dogs.copy()
y_regression = df_dogs['Stay_Duration_Days'].copy()

# 1. Temporal features
print("   Creating temporal features...")
X['Intake_Month'] = X['Intake_Date'].dt.month
X['Intake_DayOfWeek'] = X['Intake_Date'].dt.dayofweek
X['Intake_Quarter'] = X['Intake_Date'].dt.quarter
X['Intake_IsWeekend'] = (X['Intake_Date'].dt.dayofweek >= 5).astype(int)
X['Intake_DayOfMonth'] = X['Intake_Date'].dt.day
X['Intake_WeekOfYear'] = X['Intake_Date'].dt.isocalendar().week

# 2. Breed grouping
print("   Grouping rare breeds...")
breed_counts = X['Animal_Breed'].value_counts()
rare_threshold = 30
X['Breed_Group'] = X['Animal_Breed'].apply(
    lambda x: x if breed_counts.get(x, 0) >= rare_threshold else 'RARE_BREED'
)

# 3. Frequency encoding
print("   Creating frequency features...")
X['Breed_Frequency'] = X['Animal_Breed'].map(X['Animal_Breed'].value_counts())
X['Intake_Subtype_Frequency'] = X['Intake_Subtype'].map(X['Intake_Subtype'].value_counts())
X['Council_District_Frequency'] = X['Council_District'].map(X['Council_District'].value_counts())

# 4. Binary features
print("   Creating binary features...")
X['Has_Chip'] = (X['Chip_Status'].str.contains('CHIP', case=False, na=False)).astype(int)
X['Is_Stray'] = (X['Intake_Type'].str.contains('STRAY', case=False, na=False)).astype(int)
X['Is_Owner_Surrender'] = (X['Intake_Type'].str.contains('SURRENDER', case=False, na=False)).astype(int)

# Define categorical and numerical features
# CatBoost can handle categorical features natively - no encoding needed!
categorical_features = [
    'Animal_Breed',
    'Intake_Type',
    'Intake_Subtype',
    'Intake_Condition',
    'Chip_Status',
    'Animal_Origin',
    'Council_District',
    'Breed_Group'
]

numerical_features = [
    'Intake_Month',
    'Intake_DayOfWeek',
    'Intake_Quarter',
    'Intake_IsWeekend',
    'Intake_DayOfMonth',
    'Intake_WeekOfYear',
    'Breed_Frequency',
    'Intake_Subtype_Frequency',
    'Council_District_Frequency',
    'Has_Chip',
    'Is_Stray',
    'Is_Owner_Surrender'
]

all_features = categorical_features + numerical_features

# Prepare feature dataframe
print("   Preparing feature dataframe...")
X_features = X[all_features].copy()

# Handle missing values
X_features[categorical_features] = X_features[categorical_features].fillna('Unknown')
X_features[numerical_features] = X_features[numerical_features].fillna(0)

# Convert categorical features to string (CatBoost requirement)
# Some categorical features might have numeric values that need to be strings
for col in categorical_features:
    X_features[col] = X_features[col].astype(str)

# Get categorical feature indices for CatBoost
cat_indices = [X_features.columns.get_loc(col) for col in categorical_features]

print(f"   Total features: {len(all_features)}")
print(f"   - Categorical: {len(categorical_features)} (CatBoost handles natively)")
print(f"   - Numerical: {len(numerical_features)}")

# ============================================================================
# REGRESSION MODEL: Predicting Stay Duration in Days
# ============================================================================
print("\n" + "="*70)
print("REGRESSION MODEL: Predicting Stay Duration (Days)")
print("="*70)

# Remove any remaining invalid values
mask = np.isfinite(y_regression) & np.isfinite(X_features[numerical_features]).all(axis=1)
X_reg = X_features[mask].copy()
y_reg = y_regression[mask].copy()

print(f"\nDataset size: {len(X_reg)}")

# Split data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

print(f"Training set: {len(X_train_reg)}")
print(f"Test set: {len(X_test_reg)}")

# Create CatBoost Pool for efficient data handling
train_pool = Pool(
    data=X_train_reg,
    label=y_train_reg,
    cat_features=cat_indices
)

test_pool = Pool(
    data=X_test_reg,
    label=y_test_reg,
    cat_features=cat_indices
)

# CatBoost Regressor with optimized parameters
print("\nTraining CatBoost Regressor...")
print("(Using native categorical feature handling - no encoding needed!)")

catboost_reg = CatBoostRegressor(
    iterations=500,                    # Number of boosting iterations
    learning_rate=0.1,                  # Learning rate
    depth=8,                            # Tree depth (moderate to prevent overfitting)
    l2_leaf_reg=3,                      # L2 regularization
    random_strength=1,                  # Randomness for scoring
    bagging_temperature=1,              # Controls intensity of Bayesian bagging
    border_count=128,                   # Number of splits for numerical features
    loss_function='RMSE',               # Loss function
    eval_metric='R2',                   # Evaluation metric
    random_seed=42,
    verbose=100,                        # Print every 100 iterations
    early_stopping_rounds=50,           # Stop if no improvement for 50 rounds
    use_best_model=True,                # Use best model from training
    task_type='CPU'                     # Use CPU
)

# Train model
catboost_reg.fit(
    train_pool,
    eval_set=test_pool,
    plot=False
)

# Make predictions
y_train_pred_reg = catboost_reg.predict(X_train_reg)
y_test_pred_reg = catboost_reg.predict(X_test_reg)

# Calculate metrics
train_r2 = r2_score(y_train_reg, y_train_pred_reg)
test_r2 = r2_score(y_test_reg, y_test_pred_reg)
train_rmse = np.sqrt(mean_squared_error(y_train_reg, y_train_pred_reg))
test_rmse = np.sqrt(mean_squared_error(y_test_reg, y_test_pred_reg))
train_mae = mean_absolute_error(y_train_reg, y_train_pred_reg)
test_mae = mean_absolute_error(y_test_reg, y_test_pred_reg)

# Cross-validation (using Pool for proper categorical handling)
print("\nPerforming 5-fold cross-validation...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_idx, val_idx in kf.split(X_reg):
    X_cv_train, X_cv_val = X_reg.iloc[train_idx], X_reg.iloc[val_idx]
    y_cv_train, y_cv_val = y_reg.iloc[train_idx], y_reg.iloc[val_idx]
    
    train_pool_cv = Pool(X_cv_train, y_cv_train, cat_features=cat_indices)
    val_pool_cv = Pool(X_cv_val, y_cv_val, cat_features=cat_indices)
    
    model_cv = CatBoostRegressor(
        iterations=500,
        learning_rate=0.1,
        depth=8,
        l2_leaf_reg=3,
        random_seed=42,
        verbose=False,
        early_stopping_rounds=50,
        task_type='CPU'
    )
    
    model_cv.fit(train_pool_cv, eval_set=val_pool_cv)
    y_pred_cv = model_cv.predict(X_cv_val)
    cv_r2 = r2_score(y_cv_val, y_pred_cv)
    cv_scores.append(cv_r2)

cv_scores = np.array(cv_scores)

print("\n" + "="*70)
print("REGRESSION RESULTS")
print("="*70)
print(f"\nTRAINING SET:")
print(f"  R² Score:     {train_r2:.4f} ({train_r2*100:.2f}%)")
print(f"  RMSE:         {train_rmse:.2f} days")
print(f"  MAE:          {train_mae:.2f} days")

print(f"\nTEST SET - REGRESSION (LOS) PERFORMANCE:")
print(f"  R² Score:  {test_r2:.4f} ({test_r2*100:.2f}%)")
print(f"  RMSE:      {test_rmse:.2f} days")
print(f"  MAE:       {test_mae:.2f} days")

print(f"\nCROSS-VALIDATION (5-fold):")
print(f"  Mean R²:      {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)")
print(f"  Std Dev:     {cv_scores.std():.4f}")

# Feature importance
feature_importance_reg = pd.DataFrame({
    'Feature': all_features,
    'Importance': catboost_reg.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nTOP 10 MOST IMPORTANT FEATURES:")
for idx, row in feature_importance_reg.head(10).iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f} ({row['Importance']/feature_importance_reg['Importance'].sum()*100:.2f}%)")

# ============================================================================
# CLASSIFICATION MODEL: Predicting Adoption within K Days
# ============================================================================
print("\n" + "="*70)
print("CLASSIFICATION MODEL: Adopt-within-K-days")
print("="*70)

# Filter for adoption outcomes only
df_adoptions = df_dogs[df_dogs['Outcome_Type'] == 'ADOPTION'].copy()
df_adoptions = df_adoptions.dropna(subset=['Intake_Date', 'Outcome_Date', 'Stay_Duration_Days'])
df_adoptions = df_adoptions[df_adoptions['Stay_Duration_Days'] >= 0]

print(f"\nTotal adoption records: {len(df_adoptions)}")

# Prepare features for classification (same as regression)
X_clf = X_features.loc[df_adoptions.index].copy()
X_clf = X_clf[df_adoptions.index.isin(X_features.index)]

# Align indices
common_idx = X_clf.index.intersection(df_adoptions.index)
X_clf = X_clf.loc[common_idx]
df_adoptions = df_adoptions.loc[common_idx]

K_values = [7, 14, 30]
classification_results = []

for K in K_values:
    print(f"\n{'='*70}")
    print(f"CLASSIFICATION MODEL: K = {K} DAYS")
    print(f"{'='*70}")
    
    # Create binary target
    y_clf = (df_adoptions['Stay_Duration_Days'] <= K).astype(int)
    
    print(f"\nClass distribution:")
    print(f"  Adopt within {K} days (1): {y_clf.sum()} ({y_clf.sum()/len(y_clf)*100:.2f}%)")
    print(f"  Adopt after {K} days (0): {(~y_clf.astype(bool)).sum()} ({(~y_clf.astype(bool)).sum()/len(y_clf)*100:.2f}%)")
    
    # Remove invalid values
    mask_clf = np.isfinite(y_clf) & np.isfinite(X_clf[numerical_features]).all(axis=1)
    X_clf_clean = X_clf[mask_clf].copy()
    y_clf_clean = y_clf[mask_clf].copy()
    
    # Split data
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf_clean, y_clf_clean, test_size=0.3, random_state=42, stratify=y_clf_clean
    )
    
    print(f"\nTraining set: {len(X_train_clf)}")
    print(f"Test set: {len(X_test_clf)}")
    
    # Create pools
    train_pool_clf = Pool(
        data=X_train_clf,
        label=y_train_clf,
        cat_features=cat_indices
    )
    
    test_pool_clf = Pool(
        data=X_test_clf,
        label=y_test_clf,
        cat_features=cat_indices
    )
    
    # CatBoost Classifier
    print(f"\nTraining CatBoost Classifier for K={K}...")
    
    # Adjust class weights for imbalanced data
    class_weights = None
    if y_clf_clean.mean() > 0.8 or y_clf_clean.mean() < 0.2:
        # Calculate class weights for imbalanced data
        n_pos = y_clf_clean.sum()
        n_neg = len(y_clf_clean) - n_pos
        class_weights = [n_neg / n_pos, 1.0] if n_pos > 0 else None
    
    catboost_clf = CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        depth=8,
        l2_leaf_reg=3,
        random_strength=1,
        bagging_temperature=1,
        border_count=128,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50,
        use_best_model=True,
        class_weights=class_weights,  # Handle class imbalance
        task_type='CPU'
    )
    
    catboost_clf.fit(
        train_pool_clf,
        eval_set=test_pool_clf,
        plot=False
    )
    
    # Make predictions
    y_train_pred_clf = catboost_clf.predict(X_train_clf)
    y_test_pred_clf = catboost_clf.predict(X_test_clf)
    y_train_proba_clf = catboost_clf.predict_proba(X_train_clf)[:, 1]
    y_test_proba_clf = catboost_clf.predict_proba(X_test_clf)[:, 1]
    
    # Calculate metrics
    train_acc = accuracy_score(y_train_clf, y_train_pred_clf)
    test_acc = accuracy_score(y_test_clf, y_test_pred_clf)
    train_prec = precision_score(y_train_clf, y_train_pred_clf, zero_division=0)
    test_prec = precision_score(y_test_clf, y_test_pred_clf, zero_division=0)
    train_rec = recall_score(y_train_clf, y_train_pred_clf, zero_division=0)
    test_rec = recall_score(y_test_clf, y_test_pred_clf, zero_division=0)
    train_f1 = f1_score(y_train_clf, y_train_pred_clf, zero_division=0)
    test_f1 = f1_score(y_test_clf, y_test_pred_clf, zero_division=0)
    train_auc = roc_auc_score(y_train_clf, y_train_proba_clf)
    test_auc = roc_auc_score(y_test_clf, y_test_proba_clf)
    
    # Confusion matrix
    cm = confusion_matrix(y_test_clf, y_test_pred_clf)
    
    print(f"\n{'='*70}")
    print(f"CLASSIFICATION RESULTS FOR K={K} DAYS")
    print(f"{'='*70}")
    print(f"\nTRAINING SET:")
    print(f"  Accuracy:  {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Precision: {train_prec:.4f} ({train_prec*100:.2f}%)")
    print(f"  Recall:    {train_rec:.4f} ({train_rec*100:.2f}%)")
    print(f"  F1 Score:  {train_f1:.4f} ({train_f1*100:.2f}%)")
    print(f"  ROC-AUC:   {train_auc:.4f} ({train_auc*100:.2f}%)")
    
    print(f"\nTEST SET - CLASSIFICATION (K-day) PERFORMANCE:")
    print(f"  Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  ROC-AUC:   {test_auc:.4f} ({test_auc*100:.2f}%)")
    print(f"\n  Additional Metrics:")
    print(f"  Precision: {test_prec:.4f} ({test_prec*100:.2f}%)")
    print(f"  Recall:    {test_rec:.4f} ({test_rec*100:.2f}%)")
    print(f"  F1 Score:  {test_f1:.4f} ({test_f1*100:.2f}%)")
    
    # Calibration curve (Reliability curve)
    print(f"\nCALIBRATION CHECK (Reliability Curve):")
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test_clf, y_test_proba_clf, n_bins=10, strategy='uniform'
    )
    
    # Calculate calibration metrics
    calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
    print(f"  Mean Calibration Error: {calibration_error:.4f}")
    print(f"  (Lower is better - measures how well predicted probabilities match actual frequencies)")
    
    # Print calibration bins
    print(f"\n  Calibration Bins (Predicted Probability vs Actual Frequency):")
    for i in range(len(fraction_of_positives)):
        print(f"    Bin {i+1}: Predicted={mean_predicted_value[i]:.3f}, Actual={fraction_of_positives[i]:.3f}, "
              f"Diff={abs(fraction_of_positives[i] - mean_predicted_value[i]):.3f}")
    
    print(f"\nCONFUSION MATRIX (Test Set):")
    print(f"                Predicted")
    print(f"              No    Yes")
    print(f"Actual No   {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"       Yes   {cm[1,0]:5d}  {cm[1,1]:5d}")
    
    # Store results
    classification_results.append({
        'K': K,
        'Test_Accuracy': test_acc,
        'Test_Precision': test_prec,
        'Test_Recall': test_rec,
        'Test_F1': test_f1,
        'Test_ROC_AUC': test_auc,
        'Positive_Rate': y_clf_clean.mean()
    })

# Summary
print("\n" + "="*70)
print("SUMMARY: COMPARISON OF ALL MODELS")
print("="*70)

print(f"\nREGRESSION MODEL:")
print(f"  Test R² Score: {test_r2:.4f} ({test_r2*100:.2f}%)")
print(f"  Test RMSE:     {test_rmse:.2f} days")
print(f"  Test MAE:      {test_mae:.2f} days")

print(f"\nCLASSIFICATION MODELS:")
summary_df = pd.DataFrame(classification_results)
print(summary_df.to_string(index=False))

print("\n" + "="*70)
print("CATBOOST ADVANTAGES DEMONSTRATED:")
print("="*70)
print("✓ Native categorical feature handling (no encoding needed)")
print("✓ Automatic feature interactions")
print("✓ Built-in regularization and early stopping")
print("✓ Handles class imbalance automatically")
print("✓ Optimized for tabular data")
print("="*70)
print("Model training complete!")
print("="*70)
