"""
Champaign Animal Shelter Stay Duration Prediction
Using the same Random Forest and CatBoost models as Dallas data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, classification_report)
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from scipy.stats import randint
import warnings
import os
warnings.filterwarnings('ignore')

print("="*70)
print("CHAMPAIGN ANIMAL SHELTER: Stay Duration Prediction")
print("Using Random Forest and CatBoost Models")
print("="*70)

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================
print("\n1. Loading Champaign data...")

# Load all Excel files
champaign_files = [
    'Champaign/Intake and Outcome 11-1-21 through 10-31-22.xlsx',
    'Champaign/Intake and Outcome 11-1-22 through 10-31-23.xlsx',
    'Champaign/Intake and Outcome 11-1-23 through 10-31-24.xlsx'
]

dfs = []
for file in champaign_files:
    if os.path.exists(file):
        df = pd.read_excel(file)
        dfs.append(df)
        print(f"   Loaded: {file} ({len(df)} records)")

# Combine all data
df_champaign = pd.concat(dfs, ignore_index=True)
print(f"\n   Total records: {len(df_champaign)}")

# Check available columns
print(f"\n   Available columns: {list(df_champaign.columns)}")

# Filter for dogs only
print("\n2. Filtering for dogs only...")
if 'Species' in df_champaign.columns:
    df_dogs = df_champaign[df_champaign['Species'].str.upper().str.contains('DOG', na=False)].copy()
else:
    df_dogs = df_champaign.copy()
    print("   Warning: 'Species' column not found, using all records")

print(f"   Dog records: {len(df_dogs)}")

# Map Champaign columns to Dallas format
print("\n3. Mapping columns to standard format...")

# Calculate stay duration
if 'Days in Custody' in df_dogs.columns:
    df_dogs['Stay_Duration_Days'] = df_dogs['Days in Custody']
elif 'Intake Date' in df_dogs.columns and 'Outcome Date' in df_dogs.columns:
    df_dogs['Intake_Date'] = pd.to_datetime(df_dogs['Intake Date'], errors='coerce')
    df_dogs['Outcome_Date'] = pd.to_datetime(df_dogs['Outcome Date'], errors='coerce')
    df_dogs['Stay_Duration_Days'] = (df_dogs['Outcome_Date'] - df_dogs['Intake_Date']).dt.total_seconds() / (24 * 3600)
else:
    print("   Error: Cannot calculate stay duration")
    exit(1)

# Map intake date
if 'Intake Date' in df_dogs.columns:
    df_dogs['Intake_Date'] = pd.to_datetime(df_dogs['Intake Date'], errors='coerce')
elif 'Intake_Date' not in df_dogs.columns:
    print("   Error: Cannot find intake date")
    exit(1)

# Map outcome date
if 'Outcome Date' in df_dogs.columns:
    df_dogs['Outcome_Date'] = pd.to_datetime(df_dogs['Outcome Date'], errors='coerce')

# Map outcome type
if 'Outcome Type' in df_dogs.columns:
    df_dogs['Outcome_Type'] = df_dogs['Outcome Type']

# Map features to Dallas format - only use features that actually exist
column_mapping = {
    'Primary Breed': 'Animal_Breed',
    'Intake Type': 'Intake_Type',
    'Age Group at Intake': 'Intake_Condition',  # Approximate mapping
}

# Create mapped columns only if they exist
available_categorical = []
for champaign_col, dallas_col in column_mapping.items():
    if champaign_col in df_dogs.columns:
        df_dogs[dallas_col] = df_dogs[champaign_col]
        available_categorical.append(dallas_col)
    else:
        print(f"   Warning: {champaign_col} not found, skipping {dallas_col}")

print(f"\n   Available categorical features: {available_categorical}")

# Note: Champaign data doesn't have these features, so we won't use them
# Missing: Intake_Subtype, Chip_Status, Animal_Origin, Council_District

# Remove invalid records
df_dogs = df_dogs.dropna(subset=['Intake_Date', 'Stay_Duration_Days'])
df_dogs = df_dogs[df_dogs['Stay_Duration_Days'] >= 0]

print(f"\n   Valid records with stay duration: {len(df_dogs)}")

# ============================================================================
# FEATURE ENGINEERING (Same as Dallas)
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

# 2. Breed grouping (only if Animal_Breed exists)
if 'Animal_Breed' in X.columns:
    print("   Grouping rare breeds...")
    breed_counts = X['Animal_Breed'].value_counts()
    rare_threshold = 30
    X['Breed_Group'] = X['Animal_Breed'].apply(
        lambda x: x if breed_counts.get(x, 0) >= rare_threshold else 'RARE_BREED'
    )
    available_categorical.append('Breed_Group')

# 3. Frequency encoding (only for available features)
print("   Creating frequency features...")
frequency_features = []
if 'Animal_Breed' in X.columns:
    X['Breed_Frequency'] = X['Animal_Breed'].map(X['Animal_Breed'].value_counts())
    frequency_features.append('Breed_Frequency')
if 'Intake_Type' in X.columns:
    X['Intake_Type_Frequency'] = X['Intake_Type'].map(X['Intake_Type'].value_counts())
    frequency_features.append('Intake_Type_Frequency')
if 'Intake_Condition' in X.columns:
    X['Intake_Condition_Frequency'] = X['Intake_Condition'].map(X['Intake_Condition'].value_counts())
    frequency_features.append('Intake_Condition_Frequency')

# 4. Binary features (only for available features)
print("   Creating binary features...")
binary_features = []
if 'Intake_Type' in X.columns:
    X['Is_Stray'] = (X['Intake_Type'].str.contains('STRAY', case=False, na=False)).astype(int)
    binary_features.append('Is_Stray')
    X['Is_Owner_Surrender'] = (X['Intake_Type'].str.contains('SURRENDER', case=False, na=False)).astype(int)
    binary_features.append('Is_Owner_Surrender')

# Define features - only use what's available
categorical_features = [col for col in available_categorical if col in X.columns]

numerical_features = [
    'Intake_Month', 'Intake_DayOfWeek', 'Intake_Quarter', 'Intake_IsWeekend',
    'Intake_DayOfMonth', 'Intake_WeekOfYear'
] + frequency_features + binary_features

all_features = categorical_features + numerical_features

# Prepare features
X_features = X[all_features].copy()
X_features[categorical_features] = X_features[categorical_features].fillna('Unknown')
X_features[numerical_features] = X_features[numerical_features].fillna(0)

# Convert categorical to string for CatBoost
for col in categorical_features:
    X_features[col] = X_features[col].astype(str)

cat_indices = [X_features.columns.get_loc(col) for col in categorical_features]

print(f"   Total features: {len(all_features)}")
print(f"   - Categorical: {len(categorical_features)}")
print(f"   - Numerical: {len(numerical_features)}")

# Remove invalid values
mask = np.isfinite(y_regression) & np.isfinite(X_features[numerical_features]).all(axis=1)
X_clean = X_features[mask].copy()
y_clean = y_regression[mask].copy()

print(f"\n   Final dataset size: {len(X_clean)}")

# ============================================================================
# TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "="*70)
print("TRAIN-TEST SPLIT")
print("="*70)

X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_clean, test_size=0.3, random_state=42
)

print(f"Training set: {len(X_train)}")
print(f"Test set: {len(X_test)}")

# ============================================================================
# RANDOM FOREST MODEL
# ============================================================================
print("\n" + "="*70)
print("RANDOM FOREST MODEL")
print("="*70)

# Encode categorical features for Random Forest
print("Encoding categorical features for Random Forest...")
X_train_rf = X_train.copy()
X_test_rf = X_test.copy()
label_encoders = {}

for col in categorical_features:
    le = LabelEncoder()
    # Fit on combined data to handle all possible values
    all_values = pd.concat([X_train_rf[col], X_test_rf[col]]).astype(str).unique()
    le.fit(all_values)
    X_train_rf[col] = le.transform(X_train_rf[col].astype(str))
    X_test_rf[col] = le.transform(X_test_rf[col].astype(str))
    label_encoders[col] = le

# Add interaction features (only if both features exist)
if 'Intake_Type' in X_train_rf.columns and 'Intake_Condition' in X_train_rf.columns:
    X_train_rf['Intake_Type_Condition_Interaction'] = (
        X_train_rf['Intake_Type'].astype(str) + '_' + X_train_rf['Intake_Condition'].astype(str)
    )
    le_interaction = LabelEncoder()
    all_interaction_values = pd.concat([
        X_train_rf['Intake_Type_Condition_Interaction'],
        X_test_rf['Intake_Type'].astype(str) + '_' + X_test_rf['Intake_Condition'].astype(str)
    ]).unique()
    le_interaction.fit(all_interaction_values)
    X_train_rf['Intake_Type_Condition_Interaction'] = le_interaction.transform(
        X_train_rf['Intake_Type_Condition_Interaction']
    )
    X_test_rf['Intake_Type_Condition_Interaction'] = le_interaction.transform(
        X_test_rf['Intake_Type'].astype(str) + '_' + X_test_rf['Intake_Condition'].astype(str)
    )

# Use optimal hyperparameters from Dallas
print("\nTraining Random Forest with optimal hyperparameters...")
rf_model = RandomForestRegressor(
    n_estimators=223,
    max_depth=25,
    min_samples_split=9,
    min_samples_leaf=4,
    max_features='sqrt',
    bootstrap=True,
    max_samples=0.7,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_rf, y_train)

# Predictions
y_train_pred_rf = rf_model.predict(X_train_rf)
y_test_pred_rf = rf_model.predict(X_test_rf)

# Metrics
train_r2_rf = r2_score(y_train, y_train_pred_rf)
test_r2_rf = r2_score(y_test, y_test_pred_rf)
test_rmse_rf = np.sqrt(mean_squared_error(y_test, y_test_pred_rf))
test_mae_rf = mean_absolute_error(y_test, y_test_pred_rf)

print(f"\nRANDOM FOREST RESULTS (Regression - LOS):")
print(f"  R² Score:  {test_r2_rf:.4f} ({test_r2_rf*100:.2f}%)")
print(f"  RMSE:      {test_rmse_rf:.2f} days")
print(f"  MAE:       {test_mae_rf:.2f} days")
print(f"  Training R²: {train_r2_rf:.4f} ({train_r2_rf*100:.2f}%)")

# ============================================================================
# CATBOOST MODEL
# ============================================================================
print("\n" + "="*70)
print("CATBOOST MODEL")
print("="*70)

# Create pools
train_pool = Pool(X_train, y_train, cat_features=cat_indices)
test_pool = Pool(X_test, y_test, cat_features=cat_indices)

print("\nTraining CatBoost with optimal hyperparameters...")
catboost_model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.1,
    depth=8,
    l2_leaf_reg=3,
    random_strength=1,
    bagging_temperature=1,
    border_count=128,
    loss_function='RMSE',
    eval_metric='R2',
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50,
    use_best_model=True,
    task_type='CPU'
)

catboost_model.fit(train_pool, eval_set=test_pool, plot=False)

# Predictions
y_train_pred_cb = catboost_model.predict(X_train)
y_test_pred_cb = catboost_model.predict(X_test)

# Metrics
train_r2_cb = r2_score(y_train, y_train_pred_cb)
test_r2_cb = r2_score(y_test, y_test_pred_cb)
test_rmse_cb = np.sqrt(mean_squared_error(y_test, y_test_pred_cb))
test_mae_cb = mean_absolute_error(y_test, y_test_pred_cb)

print(f"\nCATBOOST RESULTS (Regression - LOS):")
print(f"  R² Score:  {test_r2_cb:.4f} ({test_r2_cb*100:.2f}%)")
print(f"  RMSE:      {test_rmse_cb:.2f} days")
print(f"  MAE:       {test_mae_cb:.2f} days")
print(f"  Training R²: {train_r2_cb:.4f} ({train_r2_cb*100:.2f}%)")

# ============================================================================
# COMPARISON SUMMARY
# ============================================================================
print("\n" + "="*70)
print("MODEL COMPARISON SUMMARY")
print("="*70)

comparison_df = pd.DataFrame({
    'Model': ['Random Forest', 'CatBoost'],
    'Test R²': [test_r2_rf, test_r2_cb],
    'Test RMSE': [test_rmse_rf, test_rmse_cb],
    'Test MAE': [test_mae_rf, test_mae_cb],
    'Train-Test Gap': [train_r2_rf - test_r2_rf, train_r2_cb - test_r2_cb]
})

print("\n" + comparison_df.to_string(index=False))

print("\n" + "="*70)
print("Champaign data modeling complete!")
print("="*70)

