"""
Visualization script for Champaign data model comparison
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(18, 12))

# Data from Champaign results (updated with latest run)
models = ['Random Forest', 'CatBoost']
test_r2 = [15.40, 13.91]  # Updated
test_rmse = [36.06, 36.38]  # Updated
test_mae = [23.27, 23.15]  # Updated
train_r2 = [39.09, 26.38]  # Updated
train_test_gap = [23.69, 12.47]  # Updated

# ============================================================================
# 1. Regression Performance Comparison
# ============================================================================
ax1 = plt.subplot(2, 3, 1)
metrics = ['R² Score (%)', 'RMSE (days)', 'MAE (days)']
rf_values = [15.40, 36.06, 23.27]
catboost_values = [13.91, 36.38, 23.15]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax1.bar(x - width/2, rf_values, width, label='Random Forest', color='#2E86AB', alpha=0.8)
bars2 = ax1.bar(x + width/2, catboost_values, width, label='CatBoost', color='#A23B72', alpha=0.8)

ax1.set_ylabel('Score / Days', fontsize=11, fontweight='bold')
ax1.set_title('Champaign: Regression Performance', fontsize=12, fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, fontsize=10, rotation=15, ha='right')
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# ============================================================================
# 2. Train-Test Performance Gap
# ============================================================================
ax2 = plt.subplot(2, 3, 2)
x = np.arange(len(models))
bars1 = ax2.bar(x - width/2, train_r2, width, label='Training R²', color='#06A77D', alpha=0.8)
bars2 = ax2.bar(x + width/2, test_r2, width, label='Test R²', color='#F18F01', alpha=0.8)

# Add gap annotation
for i, (tr, te, g) in enumerate(zip(train_r2, test_r2, train_test_gap)):
    ax2.plot([i - width/2, i + width/2], [tr, te], 'k--', linewidth=2, alpha=0.5)
    ax2.text(i, (tr + te) / 2, f'Gap: {g:.2f}%', 
            ha='center', va='center', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax2.set_ylabel('R² Score (%)', fontsize=11, fontweight='bold')
ax2.set_title('Champaign: Overfitting Control', fontsize=12, fontweight='bold', pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=10)
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)

# ============================================================================
# 3. Error Metrics Comparison
# ============================================================================
ax3 = plt.subplot(2, 3, 3)
error_metrics = ['RMSE', 'MAE']
rf_errors = [36.06, 23.27]
catboost_errors = [36.38, 23.15]

x = np.arange(len(error_metrics))
bars1 = ax3.bar(x - width/2, rf_errors, width, label='Random Forest', color='#2E86AB', alpha=0.8)
bars2 = ax3.bar(x + width/2, catboost_errors, width, label='CatBoost', color='#A23B72', alpha=0.8)

ax3.set_ylabel('Days', fontsize=11, fontweight='bold')
ax3.set_title('Champaign: Error Metrics', fontsize=12, fontweight='bold', pad=15)
ax3.set_xticks(x)
ax3.set_xticklabels(error_metrics, fontsize=10)
ax3.legend(fontsize=10)
ax3.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# ============================================================================
# 4. Comparison with Dallas Results
# ============================================================================
ax4 = plt.subplot(2, 3, 4)
datasets = ['Dallas', 'Champaign']
dallas_r2 = [38.81, 36.56]  # RF, CatBoost
champaign_r2 = [15.40, 13.91]  # RF, CatBoost (updated)

x = np.arange(len(datasets))
bars1 = ax4.bar(x - width/2, [dallas_r2[0], champaign_r2[0]], width, 
                label='Random Forest', color='#2E86AB', alpha=0.8)
bars2 = ax4.bar(x + width/2, [dallas_r2[1], champaign_r2[1]], width, 
                label='CatBoost', color='#A23B72', alpha=0.8)

ax4.set_ylabel('Test R² Score (%)', fontsize=11, fontweight='bold')
ax4.set_title('Dallas vs Champaign: R² Comparison', fontsize=12, fontweight='bold', pad=15)
ax4.set_xticks(x)
ax4.set_xticklabels(datasets, fontsize=10)
ax4.legend(fontsize=10)
ax4.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# ============================================================================
# 5. Dataset Statistics
# ============================================================================
ax5 = plt.subplot(2, 3, 5)
ax5.axis('off')

stats_text = "CHAMPAIGN DATASET STATISTICS\n" + "="*40 + "\n\n"
stats_text += f"Total Records: 6,061\n"
stats_text += f"Dog Records: 1,780\n"
stats_text += f"Training Set: 1,246 (70%)\n"
stats_text += f"Test Set: 534 (30%)\n\n"
stats_text += "FEATURES:\n"
stats_text += f"  Categorical: 4 (only available)\n"
stats_text += f"  Numerical: 11\n"
stats_text += f"  Total: 15\n\n"
stats_text += "MODELS USED:\n"
stats_text += "  • Random Forest (optimized)\n"
stats_text += "  • CatBoost (optimized)\n"
stats_text += "  • Same config as Dallas"

ax5.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', 
         facecolor='lightblue', alpha=0.5), transform=ax5.transAxes)

ax5.set_title('Dataset Information', fontsize=12, fontweight='bold', pad=15)

# ============================================================================
# 6. Key Findings Summary
# ============================================================================
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

findings_text = "KEY FINDINGS\n" + "="*40 + "\n\n"
findings_text += "✓ Random Forest slightly better\n"
findings_text += "  (R²: 15.40% vs 13.91%)\n\n"
findings_text += "✓ CatBoost better overfitting control\n"
findings_text += "  (Gap: 12.47% vs 23.69%)\n\n"
findings_text += "✓ CatBoost slightly lower MAE\n"
findings_text += "  (23.15 vs 23.27 days)\n\n"
findings_text += "✓ Lower performance than Dallas\n"
findings_text += "  (14% vs 38% R²)\n\n"
findings_text += "⚠ Smaller dataset (1,780 vs 52,748)\n"
findings_text += "  may affect model performance"

ax6.text(0.1, 0.5, findings_text, fontsize=10, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', 
         facecolor='wheat', alpha=0.5), transform=ax6.transAxes)

ax6.set_title('Key Findings', fontsize=12, fontweight='bold', pad=15)

# ============================================================================
# Overall Title
# ============================================================================
fig.suptitle('Champaign Animal Shelter: Random Forest vs CatBoost Model Comparison', 
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0.02, 1, 0.98], w_pad=2.0, h_pad=2.0)
plt.savefig('champaign_model_comparison.png', dpi=300, bbox_inches='tight', pad_inches=0.8)
print("Visualization saved as 'champaign_model_comparison.png'")
plt.show()

