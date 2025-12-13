import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, classification_report)
from scipy.stats import randint
import warnings
warnings.filterwarnings('ignore')

# Load the data
print("Loading data...")
df = pd.read_csv('Dallas_Animal_Shelter_Data_Fiscal_Year_2023_-_2026_20251115.csv')

# Filter for dogs only
print("Filtering for dogs only...")
df_dogs = df[df['Animal_Type'] == 'DOG'].copy()

# Calculate stay duration in days
print("Calculating stay duration...")
# Convert date columns to datetime
df_dogs['Intake_Date'] = pd.to_datetime(df_dogs['Intake_Date'], errors='coerce')
df_dogs['Outcome_Date'] = pd.to_datetime(df_dogs['Outcome_Date'], errors='coerce')

# Calculate duration (Outcome_Date - Intake_Date) in days
df_dogs['Stay_Duration_Days'] = (df_dogs['Outcome_Date'] - df_dogs['Intake_Date']).dt.total_seconds() / (24 * 3600)

# Filter for adoption outcomes only
print("Filtering for adoption outcomes...")
df_adoptions = df_dogs[df_dogs['Outcome_Type'] == 'ADOPTION'].copy()

# Remove rows where we can't calculate stay duration
df_adoptions = df_adoptions.dropna(subset=['Intake_Date', 'Outcome_Date', 'Stay_Duration_Days'])
df_adoptions = df_adoptions[df_adoptions['Stay_Duration_Days'] >= 0]

print(f"Total adoption records with valid stay duration: {len(df_adoptions)}")
print(f"Adoption rate: {len(df_adoptions)/len(df_dogs)*100:.2f}% of all dogs")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*60)
print("FEATURE ENGINEERING")
print("="*60)

# Prepare base features
print("Creating engineered features...")
X = df_adoptions.copy()

# 1. Temporal features from Intake_Date
print("  1. Extracting temporal features from Intake_Date...")
X['Intake_Month'] = X['Intake_Date'].dt.month
X['Intake_DayOfWeek'] = X['Intake_Date'].dt.dayofweek  # 0=Monday, 6=Sunday
X['Intake_Quarter'] = X['Intake_Date'].dt.quarter
X['Intake_IsWeekend'] = (X['Intake_DayOfWeek'] >= 5).astype(int)
X['Intake_DayOfMonth'] = X['Intake_Date'].dt.day
X['Intake_WeekOfYear'] = X['Intake_Date'].dt.isocalendar().week

# 2. Breed grouping - group rare breeds together
print("  2. Grouping rare breeds...")
breed_counts = X['Animal_Breed'].value_counts()
rare_breed_threshold = 30
X['Breed_Group'] = X['Animal_Breed'].apply(
    lambda x: x if breed_counts.get(x, 0) >= rare_breed_threshold else 'RARE_BREED'
)

# 3. Frequency encoding for high-cardinality features
print("  3. Creating frequency encoding features...")
X['Breed_Frequency'] = X['Animal_Breed'].map(X['Animal_Breed'].value_counts())
X['Intake_Subtype_Frequency'] = X['Intake_Subtype'].map(X['Intake_Subtype'].value_counts())
X['Council_District_Frequency'] = X['Council_District'].map(X['Council_District'].value_counts())

# 4. Binary features
print("  4. Creating binary features...")
X['Has_Chip'] = (X['Chip_Status'].str.contains('CHIP', case=False, na=False)).astype(int)
X['Is_Stray'] = (X['Intake_Type'].str.contains('STRAY', case=False, na=False)).astype(int)
X['Is_Owner_Surrender'] = (X['Intake_Type'].str.contains('SURRENDER', case=False, na=False)).astype(int)

# Select base categorical features
base_feature_columns = [
    'Animal_Breed',
    'Intake_Type',
    'Intake_Subtype',
    'Intake_Condition',
    'Chip_Status',
    'Animal_Origin',
    'Council_District',
    'Breed_Group'
]

# Select numerical features
numerical_feature_columns = [
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

# Combine all features
all_feature_columns = base_feature_columns + numerical_feature_columns

# Prepare feature dataframe
print("  5. Preparing feature dataframe...")
X_features = X[all_feature_columns].copy()

# Handle missing values
X_features[base_feature_columns] = X_features[base_feature_columns].fillna('Unknown')
X_features[numerical_feature_columns] = X_features[numerical_feature_columns].fillna(0)

print(f"   Total features created: {len(all_feature_columns)}")
print(f"   - Categorical features: {len(base_feature_columns)}")
print(f"   - Numerical features: {len(numerical_feature_columns)}")

# Encode categorical variables
print("\nEncoding categorical features...")
X_encoded = X_features.copy()
label_encoders = {}
for col in base_feature_columns:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
    label_encoders[col] = le

# Add interaction features
print("  6. Creating interaction features...")
X_encoded['Intake_Type_Subtype_Interaction'] = (
    X_encoded['Intake_Type'].astype(str) + '_' + X_encoded['Intake_Subtype'].astype(str)
)
le_interaction = LabelEncoder()
X_encoded['Intake_Type_Subtype_Interaction'] = le_interaction.fit_transform(
    X_encoded['Intake_Type_Subtype_Interaction']
)

X_encoded['Breed_Chip_Interaction'] = (
    X_encoded['Animal_Breed'].astype(str) + '_' + X_encoded['Chip_Status'].astype(str)
)
le_breed_chip = LabelEncoder()
X_encoded['Breed_Chip_Interaction'] = le_breed_chip.fit_transform(X_encoded['Breed_Chip_Interaction'])

# ============================================================================
# CLASSIFICATION MODELS FOR DIFFERENT K VALUES
# ============================================================================
K_values = [7, 14, 30]

print("\n" + "="*60)
print("CLASSIFICATION MODELS: Adopt-within-K-days")
print("="*60)

results_summary = []

for K in K_values:
    print(f"\n{'='*60}")
    print(f"MODEL FOR K = {K} DAYS")
    print(f"{'='*60}")
    
    # Create binary classification target: Adopt within K days?
    y = (X['Stay_Duration_Days'] <= K).astype(int)
    
    print(f"\nTarget variable distribution:")
    print(f"  Adopt within {K} days (1): {y.sum()} ({y.sum()/len(y)*100:.2f}%)")
    print(f"  Adopt after {K} days (0): {(~y.astype(bool)).sum()} ({(~y.astype(bool)).sum()/len(y)*100:.2f}%)")
    
    # Remove any remaining NaN or infinite values
    mask = np.isfinite(y) & np.isfinite(X_encoded).all(axis=1)
    X_clean = X_encoded[mask].copy()
    y_clean = y[mask].copy()
    
    # Split data: 70% training, 30% testing
    print(f"\nSplitting data: 70% training, 30% testing...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.3, random_state=42, stratify=y_clean
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Training set class distribution: {y_train.value_counts().to_dict()}")
    print(f"Test set class distribution: {y_test.value_counts().to_dict()}")
    
    # Hyperparameter tuning for Random Forest Classifier
    print(f"\nHyperparameter tuning for K={K}...")
    param_distributions = {
        'n_estimators': randint(200, 400),
        'max_depth': [10, 15, 20, 25],
        'min_samples_split': randint(5, 15),
        'min_samples_leaf': randint(3, 8),
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True],
        'max_samples': [0.7, 0.8, 0.9],
        'class_weight': [None, 'balanced']  # Handle class imbalance
    }
    
    base_rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    random_search = RandomizedSearchCV(
        base_rf,
        param_distributions,
        n_iter=20,
        cv=3,
        scoring='roc_auc',  # Use ROC-AUC for imbalanced classification
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    
    random_search.fit(X_train, y_train)
    best_rf = random_search.best_estimator_
    
    print(f"\nBest hyperparameters:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"  Best CV ROC-AUC: {random_search.best_score_:.4f} ({random_search.best_score_*100:.2f}%)")
    
    # Make predictions
    y_train_pred = best_rf.predict(X_train)
    y_test_pred = best_rf.predict(X_test)
    y_train_proba = best_rf.predict_proba(X_train)[:, 1]
    y_test_proba = best_rf.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_precision = precision_score(y_train, y_train_pred, zero_division=0)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    train_recall = recall_score(y_train, y_train_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    train_roc_auc = roc_auc_score(y_train, y_train_proba)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    print(f"\n{'='*60}")
    print(f"PERFORMANCE METRICS FOR K={K} DAYS")
    print(f"{'='*60}")
    print(f"\nTRAINING SET:")
    print(f"  Accuracy:  {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"  Precision: {train_precision:.4f} ({train_precision*100:.2f}%)")
    print(f"  Recall:    {train_recall:.4f} ({train_recall*100:.2f}%)")
    print(f"  F1 Score:  {train_f1:.4f} ({train_f1*100:.2f}%)")
    print(f"  ROC-AUC:   {train_roc_auc:.4f} ({train_roc_auc*100:.2f}%)")
    
    print(f"\nTEST SET:")
    print(f"  Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"  Precision: {test_precision:.4f} ({test_precision*100:.2f}%)")
    print(f"  Recall:    {test_recall:.4f} ({test_recall*100:.2f}%)")
    print(f"  F1 Score:  {test_f1:.4f} ({test_f1*100:.2f}%)")
    print(f"  ROC-AUC:   {test_roc_auc:.4f} ({test_roc_auc*100:.2f}%)")
    
    print(f"\nCONFUSION MATRIX (Test Set):")
    print(f"                Predicted")
    print(f"              No    Yes")
    print(f"Actual No   {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"       Yes   {cm[1,0]:5d}  {cm[1,1]:5d}")
    
    # Classification report
    print(f"\nCLASSIFICATION REPORT (Test Set):")
    print(classification_report(y_test, y_test_pred, target_names=['After K days', 'Within K days']))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns.tolist(),
        'Importance': best_rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\nTOP 10 MOST IMPORTANT FEATURES:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f} ({row['Importance']*100:.2f}%)")
    
    # Store results
    results_summary.append({
        'K': K,
        'Test_Accuracy': test_accuracy,
        'Test_Precision': test_precision,
        'Test_Recall': test_recall,
        'Test_F1': test_f1,
        'Test_ROC_AUC': test_roc_auc,
        'Positive_Class_Rate': y_clean.mean()
    })

# Summary comparison
print("\n" + "="*60)
print("SUMMARY COMPARISON: All K Values")
print("="*60)
summary_df = pd.DataFrame(results_summary)
print(summary_df.to_string(index=False))

print("\n" + "="*60)
print("Model training and evaluation complete!")
print("="*60)

