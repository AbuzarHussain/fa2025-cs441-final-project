import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import randint, uniform
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

# Remove rows where we can't calculate stay duration (missing dates or negative duration)
df_dogs = df_dogs.dropna(subset=['Intake_Date', 'Outcome_Date', 'Stay_Duration_Days'])
df_dogs = df_dogs[df_dogs['Stay_Duration_Days'] >= 0]

print(f"Total dog records with valid stay duration: {len(df_dogs)}")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*60)
print("FEATURE ENGINEERING")
print("="*60)

# Prepare base features
print("Creating engineered features...")
X = df_dogs.copy()
y = df_dogs['Stay_Duration_Days'].copy()

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
rare_breed_threshold = 30  # Breeds with <30 occurrences are grouped
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

# 5. Interaction features (combinations of important features)
print("  5. Creating interaction features...")
# These will be created after encoding, so we'll add them later

# Select base categorical features
base_feature_columns = [
    'Animal_Breed',
    'Intake_Type',
    'Intake_Subtype',
    'Intake_Condition',
    'Chip_Status',
    'Animal_Origin',
    'Council_District',
    'Breed_Group'  # New engineered feature
]

# Select numerical features (temporal and frequency)
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
print("  6. Preparing feature dataframe...")
X_features = X[all_feature_columns].copy()

# Handle missing values
X_features[base_feature_columns] = X_features[base_feature_columns].fillna('Unknown')
X_features[numerical_feature_columns] = X_features[numerical_feature_columns].fillna(0)

print(f"   Total features created: {len(all_feature_columns)}")
print(f"   - Categorical features: {len(base_feature_columns)}")
print(f"   - Numerical features: {len(numerical_feature_columns)}")

# Update feature_columns for encoding
feature_columns = base_feature_columns

# Encode categorical variables - try both label encoding and one-hot encoding
print("\nEncoding categorical features...")
print("Testing both Label Encoding and One-Hot Encoding...")

# Method 1: Label Encoding (original method)
X_label = X_features.copy()
label_encoders = {}
for col in feature_columns:
    le = LabelEncoder()
    X_label[col] = le.fit_transform(X_label[col].astype(str))
    label_encoders[col] = le

# Add interaction features after encoding
print("  7. Creating interaction features...")
# Interaction: Intake_Type × Intake_Subtype (important interaction)
X_label['Intake_Type_Subtype_Interaction'] = (
    X_label['Intake_Type'].astype(str) + '_' + X_label['Intake_Subtype'].astype(str)
)
le_interaction = LabelEncoder()
X_label['Intake_Type_Subtype_Interaction'] = le_interaction.fit_transform(
    X_label['Intake_Type_Subtype_Interaction']
)

# Interaction: Breed × Chip_Status
X_label['Breed_Chip_Interaction'] = (
    X_label['Animal_Breed'].astype(str) + '_' + X_label['Chip_Status'].astype(str)
)
le_breed_chip = LabelEncoder()
X_label['Breed_Chip_Interaction'] = le_breed_chip.fit_transform(X_label['Breed_Chip_Interaction'])

# Method 2: One-Hot Encoding (can help with categorical features)
X_onehot = pd.get_dummies(X_features, columns=feature_columns, prefix=feature_columns, drop_first=False)

# Add interaction features for one-hot encoding
X_onehot['Intake_Type_Subtype_Interaction'] = (
    X_onehot.filter(regex='Intake_Type_').idxmax(axis=1).str.replace('Intake_Type_', '') + '_' +
    X_onehot.filter(regex='Intake_Subtype_').idxmax(axis=1).str.replace('Intake_Subtype_', '')
)
X_onehot = pd.get_dummies(X_onehot, columns=['Intake_Type_Subtype_Interaction'], prefix='Interaction', drop_first=False)

# Remove any remaining NaN or infinite values
mask_label = np.isfinite(y) & np.isfinite(X_label).all(axis=1)
mask_onehot = np.isfinite(y) & np.isfinite(X_onehot).all(axis=1)

X_label = X_label[mask_label]
X_onehot = X_onehot[mask_onehot]
y_label = y[mask_label]
y_onehot = y[mask_onehot]

print(f"Final dataset size (Label Encoding): {len(X_label)}")
print(f"Final dataset size (One-Hot Encoding): {len(X_onehot)}")

# Split data: 70% training, 30% testing
print("\nSplitting data: 70% training, 30% testing...")
X_train_label, X_test_label, y_train_label, y_test_label = train_test_split(
    X_label, y_label, test_size=0.3, random_state=42
)
X_train_onehot, X_test_onehot, y_train_onehot, y_test_onehot = train_test_split(
    X_onehot, y_onehot, test_size=0.3, random_state=42
)

print(f"Training set size (Label): {len(X_train_label)}")
print(f"Test set size (Label): {len(X_test_label)}")
print(f"Training set size (One-Hot): {len(X_train_onehot)}")
print(f"Test set size (One-Hot): {len(X_test_onehot)}")

# Hyperparameter tuning for Random Forest
print("\n" + "="*60)
print("HYPERPARAMETER TUNING")
print("="*60)
print("Performing randomized search for optimal hyperparameters...")

# Define parameter grid for hyperparameter tuning
# Adjusted to prevent overfitting with stronger regularization
param_distributions = {
    'n_estimators': randint(200, 400),  # 200-400 trees (reduced from 500)
    'max_depth': [10, 15, 20, 25],  # Limited depth to prevent overfitting (removed 30 and None)
    'min_samples_split': randint(5, 15),  # Increased from 2-10 to 5-15 for more regularization
    'min_samples_leaf': randint(3, 8),  # Increased from 1-5 to 3-8 for more regularization
    'max_features': ['sqrt', 'log2'],  # Removed None to force feature subset selection
    'bootstrap': [True],  # Always use bootstrap for better generalization
    'max_samples': [0.7, 0.8, 0.9]  # Use subset of samples (removed None)
}

# Base model for tuning
base_rf = RandomForestRegressor(random_state=42, n_jobs=-1)

# Randomized search with cross-validation (using label encoding first)
# Reduced iterations and CV folds for faster execution
print("\nTuning with Label Encoding...")
print("This will take a few minutes... (20 iterations × 3-fold CV = 60 fits)")
random_search_label = RandomizedSearchCV(
    base_rf,
    param_distributions,
    n_iter=20,  # Reduced from 50 for faster execution
    cv=3,  # Reduced from 5 for faster execution
    scoring='r2',
    n_jobs=-1,
    random_state=42,
    verbose=1
)
random_search_label.fit(X_train_label, y_train_label)

print("\nTuning with One-Hot Encoding...")
print("This will take a few minutes... (20 iterations × 3-fold CV = 60 fits)")
random_search_onehot = RandomizedSearchCV(
    base_rf,
    param_distributions,
    n_iter=20,  # Reduced from 50 for faster execution
    cv=3,  # Reduced from 5 for faster execution
    scoring='r2',
    n_jobs=-1,
    random_state=42,
    verbose=1
)
random_search_onehot.fit(X_train_onehot, y_train_onehot)

# Get best models
best_rf_label = random_search_label.best_estimator_
best_rf_onehot = random_search_onehot.best_estimator_

print("\n" + "="*60)
print("BEST HYPERPARAMETERS (Label Encoding)")
print("="*60)
for param, value in random_search_label.best_params_.items():
    print(f"  {param}: {value}")
print(f"  Best CV Score (R²): {random_search_label.best_score_:.4f} ({random_search_label.best_score_*100:.2f}%)")

print("\n" + "="*60)
print("BEST HYPERPARAMETERS (One-Hot Encoding)")
print("="*60)
for param, value in random_search_onehot.best_params_.items():
    print(f"  {param}: {value}")
print(f"  Best CV Score (R²): {random_search_onehot.best_score_:.4f} ({random_search_onehot.best_score_*100:.2f}%)")

# Choose the best encoding method based on CV score
if random_search_label.best_score_ >= random_search_onehot.best_score_:
    print("\n✓ Using Label Encoding (better CV performance)")
    rf_model = best_rf_label
    X_train, X_test, y_train, y_test = X_train_label, X_test_label, y_train_label, y_test_label
    encoding_method = "Label Encoding"
else:
    print("\n✓ Using One-Hot Encoding (better CV performance)")
    rf_model = best_rf_onehot
    X_train, X_test, y_train, y_test = X_train_onehot, X_test_onehot, y_train_onehot, y_test_onehot
    encoding_method = "One-Hot Encoding"

# Additional cross-validation evaluation
print("\n" + "="*60)
print("CROSS-VALIDATION EVALUATION")
print("="*60)
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=3, scoring='r2', n_jobs=-1)
print(f"  3-Fold CV R² Scores: {cv_scores}")
print(f"  Mean CV R² Score: {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)")
print(f"  CV R² Std Dev: {cv_scores.std():.4f}")

# Make predictions
print("Making predictions...")
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# Calculate metrics
print("\n" + "="*60)
print("MODEL PERFORMANCE METRICS")
print("="*60)

# Training set metrics
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)

# Test set metrics
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Display accuracy prominently
print("\n" + "="*60)
print("ACCURACY SUMMARY")
print("="*60)
print(f"  TRAIN ACCURACY (R² Score): {train_r2*100:.2f}%")
print(f"  TEST ACCURACY (R² Score):  {test_r2*100:.2f}%")
print("="*60)

print("\nTRAINING SET (70%):")
print(f"  Mean Squared Error (MSE): {train_mse:.2f}")
print(f"  Root Mean Squared Error (RMSE): {train_rmse:.2f} days")
print(f"  Mean Absolute Error (MAE): {train_mae:.2f} days")
print(f"  R² Score (Accuracy): {train_r2:.4f} ({train_r2*100:.2f}%)")

print("\nTEST SET (30%):")
print(f"  Mean Squared Error (MSE): {test_mse:.2f}")
print(f"  Root Mean Squared Error (RMSE): {test_rmse:.2f} days")
print(f"  Mean Absolute Error (MAE): {test_mae:.2f} days")
print(f"  R² Score (Accuracy): {test_r2:.4f} ({test_r2*100:.2f}%)")


print("\n" + "="*60)
print("TOP 15 MOST IMPORTANT FEATURES")
print("="*60)

feature_names = X_train.columns.tolist()

feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

for idx, row in feature_importance.head(15).iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f} ({row['Importance']*100:.2f}%)")

print("\n" + "="*60)
print("TARGET VARIABLE STATISTICS")
print("="*60)
print(f"  Mean stay duration: {y.mean():.2f} days")
print(f"  Median stay duration: {y.median():.2f} days")
print(f"  Min stay duration: {y.min():.2f} days")
print(f"  Max stay duration: {y.max():.2f} days")
print(f"  Std deviation: {y.std():.2f} days")
print("\n" + "="*60)
print("IMPROVEMENTS MADE")
print("="*60)
print("  FEATURE ENGINEERING:")
print("    1. Temporal features: Month, DayOfWeek, Quarter, IsWeekend, DayOfMonth, WeekOfYear")
print("    2. Breed grouping: Grouped rare breeds (<30 occurrences)")
print("    3. Frequency encoding: Breed, Intake_Subtype, Council_District frequencies")
print("    4. Binary features: Has_Chip, Is_Stray, Is_Owner_Surrender")
print("    5. Interaction features: Intake_Type×Subtype, Breed×Chip_Status")
print("  MODEL OPTIMIZATION:")
print("    6. Hyperparameter tuning using RandomizedSearchCV (20 iterations)")
print("    7. Tested both Label Encoding and One-Hot Encoding")
print("    8. 3-fold cross-validation for robust evaluation")
print("    9. Optimized: n_estimators, max_depth, min_samples_split, min_samples_leaf")
print("   10. Optimized: max_features, bootstrap, max_samples")
print(f"   11. Selected encoding method: {encoding_method}")
print(f"   12. Best hyperparameters found through systematic search")

print("\n" + "="*60)
print("Model training and evaluation complete!")
print("="*60)
