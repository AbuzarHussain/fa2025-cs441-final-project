"""
Visualization script for comparing Random Forest and CatBoost models
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(20, 14))

# ============================================================================
# 1. Regression Performance Comparison
# ============================================================================
ax1 = plt.subplot(2, 3, 1)
metrics = ['R² Score', 'RMSE (days)', 'MAE (days)']
rf_regression = [38.81, 12.73, 6.80]
catboost_regression = [36.56, 12.70, 6.82]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax1.bar(x - width/2, rf_regression, width, label='Random Forest', color='#2E86AB', alpha=0.8)
bars2 = ax1.bar(x + width/2, catboost_regression, width, label='CatBoost', color='#A23B72', alpha=0.8)

ax1.set_ylabel('Score / Days', fontsize=11, fontweight='bold')
ax1.set_title('Regression Performance Comparison', fontsize=12, fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, fontsize=10)
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# ============================================================================
# 2. Classification Performance (ROC-AUC)
# ============================================================================
ax2 = plt.subplot(2, 3, 2)
k_values = ['K=7', 'K=14', 'K=30']
rf_roc_auc = [80.71, 76.78, 73.41]
catboost_roc_auc = [79.39, 76.64, 70.17]

x = np.arange(len(k_values))
bars1 = ax2.bar(x - width/2, rf_roc_auc, width, label='Random Forest', color='#2E86AB', alpha=0.8)
bars2 = ax2.bar(x + width/2, catboost_roc_auc, width, label='CatBoost', color='#A23B72', alpha=0.8)

ax2.set_ylabel('ROC-AUC (%)', fontsize=11, fontweight='bold')
ax2.set_title('Classification ROC-AUC Comparison', fontsize=12, fontweight='bold', pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(k_values, fontsize=10)
ax2.legend(fontsize=10)
ax2.set_ylim([65, 85])
ax2.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# ============================================================================
# 3. Classification Accuracy Comparison
# ============================================================================
ax3 = plt.subplot(2, 3, 3)
rf_accuracy = [75.32, 83.83, 94.67]
catboost_accuracy = [73.92, 83.40, 94.67]

bars1 = ax3.bar(x - width/2, rf_accuracy, width, label='Random Forest', color='#2E86AB', alpha=0.8)
bars2 = ax3.bar(x + width/2, catboost_accuracy, width, label='CatBoost', color='#A23B72', alpha=0.8)

ax3.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax3.set_title('Classification Accuracy Comparison', fontsize=12, fontweight='bold', pad=15)
ax3.set_xticks(x)
ax3.set_xticklabels(k_values, fontsize=10)
ax3.legend(fontsize=10)
ax3.set_ylim([70, 100])
ax3.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# ============================================================================
# 4. Feature Importance Comparison (Top 5)
# ============================================================================
ax4 = plt.subplot(2, 3, 4)
# Random Forest top features
rf_features = ['Intake_Subtype', 'Animal_Origin', 'Chip_Status', 'Intake_Condition', 'Intake_Type']
rf_importance = [26.27, 23.45, 18.84, 10.29, 7.28]

# CatBoost top features
catboost_features = ['Chip_Status', 'Intake_Type', 'Intake_Condition', 'Intake_Subtype', 'Animal_Origin']
catboost_importance = [30.66, 21.54, 10.42, 9.10, 5.05]

y_pos = np.arange(len(rf_features))
bars1 = ax4.barh(y_pos - width/2, rf_importance, width, label='Random Forest', color='#2E86AB', alpha=0.8)
bars2 = ax4.barh(y_pos + width/2, catboost_importance, width, label='CatBoost', color='#A23B72', alpha=0.8)

ax4.set_xlabel('Importance (%)', fontsize=11, fontweight='bold')
ax4.set_title('Top 5 Feature Importance Comparison', fontsize=12, fontweight='bold', pad=15)
ax4.set_yticks(y_pos)
ax4.set_yticklabels(rf_features, fontsize=9)
ax4.legend(fontsize=10)
ax4.grid(axis='x', alpha=0.3)

# ============================================================================
# 5. Overfitting Control (Train-Test Gap)
# ============================================================================
ax5 = plt.subplot(2, 3, 5)
models = ['Random Forest', 'CatBoost']
train_r2 = [51.46, 38.42]
test_r2 = [38.81, 36.56]
gap = [12.65, 1.86]

x = np.arange(len(models))
bars1 = ax5.bar(x - width/2, train_r2, width, label='Training R²', color='#06A77D', alpha=0.8)
bars2 = ax5.bar(x + width/2, test_r2, width, label='Test R²', color='#F18F01', alpha=0.8)

# Add gap annotation
for i, (tr, te, g) in enumerate(zip(train_r2, test_r2, gap)):
    ax5.plot([i - width/2, i + width/2], [tr, te], 'k--', linewidth=2, alpha=0.5)
    ax5.text(i, (tr + te) / 2, f'Gap: {g:.2f}%', 
            ha='center', va='center', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax5.set_ylabel('R² Score (%)', fontsize=11, fontweight='bold')
ax5.set_title('Overfitting Control Comparison', fontsize=12, fontweight='bold', pad=15)
ax5.set_xticks(x)
ax5.set_xticklabels(models, fontsize=10)
ax5.legend(fontsize=10)
ax5.grid(axis='y', alpha=0.3)

# ============================================================================
# 6. Advantages/Disadvantages Summary
# ============================================================================
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

# Create comparison table with shorter text
comparison_data = {
    'Aspect': ['Regression R²', 'ROC-AUC', 'Preprocessing', 
               'Hyperparameter', 'Early Stop', 'Feature Imp'],
    'Random Forest': ['✓ 38.81%', '✓ Better', 
                     '✗ Encoding', '✗ Extensive',
                     '✗ Manual', '✓ Intuitive'],
    'CatBoost': ['✗ 36.56%', '✗ Lower',
                '✓ Native', '✓ Less',
                '✓ Auto', '✗ Less']
}

# Create text table with shorter format
table_text = "MODEL COMPARISON\n" + "="*40 + "\n\n"
table_text += f"{'Aspect':<18} {'RF':<12} {'CatBoost':<12}\n"
table_text += "-"*42 + "\n"

for i, aspect in enumerate(comparison_data['Aspect']):
    rf_val = comparison_data['Random Forest'][i]
    cb_val = comparison_data['CatBoost'][i]
    table_text += f"{aspect:<18} {rf_val:<12} {cb_val:<12}\n"

ax6.text(0.1, 0.5, table_text, fontsize=8.5, family='monospace',
         verticalalignment='center', horizontalalignment='left',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
         transform=ax6.transAxes)

ax6.set_title('Key Differences Summary', fontsize=12, fontweight='bold', pad=15)

# ============================================================================
# Overall Title
# ============================================================================
fig.suptitle('Random Forest vs CatBoost: Comprehensive Model Comparison', 
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0.02, 1, 0.98], w_pad=2.0, h_pad=2.0)
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight', pad_inches=0.8)
print("Visualization saved as 'model_comparison.png'")
plt.show()

