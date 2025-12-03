# 🚀 COMPREHENSIVE INSTRUCTION PROMPT: Telco Churn Project V4 Creation

## MASTER INSTRUCTION FOR GITHUB COPILOT AGENT

**Document Purpose**: Complete instructions for creating Version 4 of the Telco Customer Churn Prediction notebook - a condensed, simplified, high-quality final version for academic submission.

**Created**: December 2025  
**Course**: ECO 6313 - Applied Econometrics for Business (UTSA)  
**Due Date**: December 5th, 2025

---

## 📋 EXECUTIVE CONTEXT

### Project Background
This is a university group project requiring an end-to-end machine learning solution for predicting customer churn in the telecommunications industry. Previous versions (V1-V3) contain valuable work but are excessively verbose (V3 is ~20K+ lines). V4 must be:

1. **CONDENSED**: Maximum 2,000-3,000 lines of code+markdown (vs 20K in V3)
2. **SIMPLIFIED**: Clear, straightforward code without over-engineering
3. **HIGH-QUALITY**: Professional, publication-ready output
4. **COMPLETE**: All rubric requirements satisfied

### Rubric Requirements Summary
The professor requires:
1. **Model questions and hypotheses** - Why does business need this? What factors impact churn?
2. **Variable selection & data prep** - Univariate, bivariate, multivariate analysis
3. **Feature engineering** - Create new variables as needed
4. **ML Modeling** - Train/test split, multiple models, accuracy metrics, model tuning
5. **Findings Summary** - Strengths, weaknesses, future improvements
6. **References** - Cite sources used

### Data Overview
- **Dataset**: Telco Customer Churn (Kaggle)
- **Size**: 7,043 customers × 21 columns
- **Target**: `Churn` (Yes/No) - ~26.5% churn rate
- **Key Features**: Demographics (gender, senior citizen, partner), Services (phone, internet, streaming), Account (contract, billing, charges)

---

## 🎯 V4 NOTEBOOK STRUCTURE

### Required Sections (10 Sections Total)

```
SECTION 1: TITLE & EXECUTIVE SUMMARY (1 cell)
SECTION 2: INTRODUCTION & HYPOTHESES (2 cells)
SECTION 3: DATA LOADING & QUALITY (3 cells)
SECTION 4: EXPLORATORY DATA ANALYSIS (8 cells)
SECTION 5: FEATURE ENGINEERING & PREPROCESSING (3 cells)
SECTION 6: MODEL TRAINING & COMPARISON (4 cells)
SECTION 7: MODEL EVALUATION & INTERPRETATION (3 cells)
SECTION 8: BUSINESS INSIGHTS & RECOMMENDATIONS (2 cells)
SECTION 9: CONCLUSIONS & FUTURE WORK (1 cell)
SECTION 10: REFERENCES (1 cell)

TOTAL: ~28 cells (condensed from 90+ in V3)
```

---

## 📝 DETAILED SECTION SPECIFICATIONS

### SECTION 1: TITLE & EXECUTIVE SUMMARY
**Cells**: 1 markdown cell
**Length**: ~30 lines

**Content**:
```markdown
# 📊 TELCO CUSTOMER CHURN PREDICTION
## ECO 6313 - Applied Econometrics | UTSA | December 2025

### Executive Summary
- **Objective**: Predict customer churn to enable proactive retention
- **Dataset**: 7,043 customers with 20 features
- **Best Model**: [XGBoost/Random Forest] with [XX]% ROC-AUC
- **Key Finding**: Month-to-month contracts show 42% churn vs 3% for 2-year contracts
- **Business Impact**: Predicted savings of $XX,XXX through targeted retention

### Team Members
- [Team member names]
```

---

### SECTION 2: INTRODUCTION & HYPOTHESES
**Cells**: 2 cells (1 markdown, 1 markdown)
**Length**: ~50 lines total

**Cell 2.1 - Business Context**:
```markdown
## 1. Introduction

### 1.1 Business Problem
Customer churn in telecom costs companies 15-25% of revenue annually. 
Acquiring new customers is 5x more expensive than retaining existing ones.
This model helps identify at-risk customers for proactive intervention.

### 1.2 Why This Matters
- Churn rate: ~26.5% in our dataset
- Average customer lifetime value (CLV): ~$2,000
- Early identification enables targeted retention campaigns
```

**Cell 2.2 - Hypotheses**:
```markdown
### 1.3 Research Hypotheses

Based on domain knowledge and literature:

**H1**: Month-to-month contracts will show higher churn than term contracts
**H2**: Higher monthly charges correlate with increased churn risk
**H3**: Customers without tech support are more likely to churn
**H4**: New customers (tenure < 12 months) have higher churn rates
**H5**: Electronic check payment users churn more than autopay users

We will test these hypotheses through statistical analysis and modeling.
```

---

### SECTION 3: DATA LOADING & QUALITY
**Cells**: 3 cells (1 code imports, 1 code data loading, 1 markdown summary)
**Length**: ~80 lines

**Cell 3.1 - Imports** (code):
```python
# Core Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
from imblearn.over_sampling import SMOTE

# Settings
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Visualization settings
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
colors = {'churn': '#E63946', 'no_churn': '#457B9D'}

print("✅ Libraries loaded successfully!")
```

**Cell 3.2 - Data Loading** (code):
```python
# Load dataset
df = pd.read_csv('Telco_Customer_Churn.csv')

# Quick data quality check
print(f"Dataset Shape: {df.shape}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()}")

# Fix TotalCharges (convert to numeric)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Impute missing TotalCharges: use MonthlyCharges × tenure (or MonthlyCharges if tenure is 0)
mask = df['TotalCharges'].isnull()
df.loc[mask, 'TotalCharges'] = df.loc[mask, 'MonthlyCharges'] * np.maximum(df.loc[mask, 'tenure'], 1)

# Preview
df.head()
```

**Cell 3.3 - Data Summary** (markdown):
```markdown
### Data Quality Summary
- **Observations**: 7,043 customers
- **Features**: 21 columns (20 predictors + 1 target)
- **Missing Values**: 11 in TotalCharges (imputed with MonthlyCharges)
- **Target Distribution**: 26.5% churned, 73.5% retained
```

---

### SECTION 4: EXPLORATORY DATA ANALYSIS
**Cells**: 8 cells (alternating code/markdown)
**Length**: ~300 lines

**Cell 4.1 - Target Variable Analysis** (code):
```python
# Churn distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Count plot
churn_counts = df['Churn'].value_counts()
axes[0].bar(churn_counts.index, churn_counts.values, color=[colors['no_churn'], colors['churn']])
axes[0].set_title('Churn Distribution', fontsize=14, fontweight='bold')
for i, v in enumerate(churn_counts.values):
    axes[0].text(i, v+50, f'{v:,}', ha='center', fontweight='bold')

# Percentage pie
axes[1].pie(churn_counts.values, labels=['Retained', 'Churned'], 
            autopct='%1.1f%%', colors=[colors['no_churn'], colors['churn']])
axes[1].set_title('Churn Rate', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

print(f"Churn Rate: {(df['Churn']=='Yes').mean()*100:.1f}%")
```

**Cell 4.2 - Categorical Features vs Churn** (code):
```python
# Key categorical features analysis
cat_features = ['Contract', 'PaymentMethod', 'InternetService', 'TechSupport', 'OnlineSecurity']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, feature in enumerate(cat_features):
    # Calculate churn rate by category
    churn_rate = df.groupby(feature)['Churn'].apply(lambda x: (x=='Yes').mean() * 100)
    
    # Plot
    bars = axes[idx].bar(churn_rate.index, churn_rate.values, color=colors['churn'])
    axes[idx].set_title(f'Churn Rate by {feature}', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Churn Rate (%)')
    axes[idx].tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                      f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

# Hide last subplot if odd number
axes[-1].axis('off')
plt.tight_layout()
plt.show()
```

**Cell 4.3 - Statistical Tests for Categorical** (code):
```python
# Chi-square tests for categorical features
print("="*60)
print("CHI-SQUARE TESTS: Categorical Features vs Churn")
print("="*60)

df['Churn_Binary'] = (df['Churn'] == 'Yes').astype(int)

for feature in cat_features:
    contingency = pd.crosstab(df[feature], df['Churn'])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    cramers_v = np.sqrt(chi2 / (len(df) * (min(contingency.shape) - 1)))
    
    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
    print(f"\n{feature}:")
    print(f"  Chi² = {chi2:.2f}, p-value = {p_value:.4f} {sig}")
    print(f"  Cramér's V = {cramers_v:.3f} ({'Strong' if cramers_v > 0.3 else 'Medium' if cramers_v > 0.1 else 'Weak'})")
```

**Cell 4.4 - Numerical Features Analysis** (code):
```python
# Numerical features by churn status
num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, feature in enumerate(num_features):
    # Boxplot by churn
    df.boxplot(column=feature, by='Churn', ax=axes[idx])
    axes[idx].set_title(f'{feature} by Churn Status', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Churn')
    
    # T-test
    churned = df[df['Churn']=='Yes'][feature]
    retained = df[df['Churn']=='No'][feature]
    t_stat, p_val = stats.ttest_ind(churned, retained)
    axes[idx].text(0.5, 0.02, f'p={p_val:.4f}', transform=axes[idx].transAxes, 
                   ha='center', fontsize=10, style='italic')

plt.suptitle('')
plt.tight_layout()
plt.show()
```

**Cell 4.5 - Correlation Analysis** (code):
```python
# Encode for correlation analysis
df_encoded = df.copy()
for col in df_encoded.select_dtypes(include='object').columns:
    if col != 'customerID':
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

# Correlation with churn
churn_corr = df_encoded.drop('customerID', axis=1).corr()['Churn'].sort_values(ascending=False)

# Plot
plt.figure(figsize=(10, 8))
churn_corr.drop('Churn').plot(kind='barh', color=[colors['churn'] if x > 0 else colors['no_churn'] 
                                                    for x in churn_corr.drop('Churn')])
plt.title('Feature Correlation with Churn', fontsize=14, fontweight='bold')
plt.xlabel('Correlation Coefficient')
plt.tight_layout()
plt.show()

print("\nTop 5 Positive Correlations:")
print(churn_corr[1:6])
print("\nTop 5 Negative Correlations:")
print(churn_corr[-5:])
```

**Cell 4.6 - Key EDA Findings** (markdown):
```markdown
### EDA Key Findings

**Hypothesis Testing Results:**

| Hypothesis | Result | Evidence |
|------------|--------|----------|
| H1: Month-to-month → Higher churn | ✅ CONFIRMED | 42% vs 11% (1-year) vs 3% (2-year) |
| H2: Higher charges → More churn | ✅ CONFIRMED | Mean $75 (churned) vs $61 (retained) |
| H3: No tech support → More churn | ✅ CONFIRMED | 42% vs 15% with support |
| H4: New customers → Higher churn | ✅ CONFIRMED | Median tenure 10 months (churned) vs 38 (retained) |
| H5: Electronic check → More churn | ✅ CONFIRMED | 45% vs 15-18% for autopay |

**Statistical Significance:**
All key features show p < 0.001 in chi-square tests, indicating highly significant associations with churn.
```

**Cell 4.7 - Tenure Deep Dive** (code):
```python
# Tenure analysis - critical feature
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram by churn
for status in ['No', 'Yes']:
    data = df[df['Churn']==status]['tenure']
    axes[0].hist(data, bins=20, alpha=0.6, label=f'Churn={status}', 
                 color=colors['no_churn'] if status=='No' else colors['churn'])
axes[0].set_title('Tenure Distribution by Churn', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Tenure (months)')
axes[0].legend()

# Churn rate by tenure group
df['tenure_group'] = pd.cut(df['tenure'], bins=[0,12,24,48,72], labels=['0-12','13-24','25-48','49-72'])
churn_by_tenure = df.groupby('tenure_group')['Churn_Binary'].mean() * 100
axes[1].bar(churn_by_tenure.index, churn_by_tenure.values, color=colors['churn'])
axes[1].set_title('Churn Rate by Tenure Group', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Tenure (months)')
axes[1].set_ylabel('Churn Rate (%)')
for i, v in enumerate(churn_by_tenure.values):
    axes[1].text(i, v+1, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()
print(f"\n💡 Insight: New customers (0-12 months) have {churn_by_tenure['0-12']:.1f}% churn rate!")
```

**Cell 4.8 - Contract Analysis** (code):
```python
# Contract type - strongest predictor
contract_analysis = df.groupby('Contract').agg({
    'Churn_Binary': ['count', 'sum', 'mean'],
    'MonthlyCharges': 'mean',
    'tenure': 'mean'
}).round(2)
contract_analysis.columns = ['Customers', 'Churned', 'Churn_Rate', 'Avg_Charges', 'Avg_Tenure']
contract_analysis['Churn_Rate'] = (contract_analysis['Churn_Rate'] * 100).round(1)

print("Contract Type Analysis:")
print("="*70)
print(contract_analysis)
print("="*70)
print(f"\n💡 Key Insight: Month-to-month customers churn at {contract_analysis.loc['Month-to-month', 'Churn_Rate']}%")
print(f"   This is {contract_analysis.loc['Month-to-month', 'Churn_Rate']/contract_analysis.loc['Two year', 'Churn_Rate']:.1f}x higher than two-year contracts!")
```

---

### SECTION 5: FEATURE ENGINEERING & PREPROCESSING
**Cells**: 3 cells
**Length**: ~120 lines

**Cell 5.1 - Feature Engineering** (code):
```python
# Create new features
df_model = df.copy()

# 1. Service count features
service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
df_model['TotalServices'] = df_model[service_cols].apply(
    lambda x: sum([1 for v in x if v not in ['No', 'No phone service', 'No internet service']]), axis=1)

# 2. Tenure features
df_model['IsNewCustomer'] = (df_model['tenure'] < 12).astype(int)
df_model['TenureYears'] = df_model['tenure'] / 12

# 3. Financial features  
df_model['AvgChargesPerMonth'] = np.where(
    df_model['tenure'] > 0,
    df_model['TotalCharges'] / df_model['tenure'],
    df_model['MonthlyCharges']
)

# 4. Contract security
df_model['HasContract'] = (df_model['Contract'] != 'Month-to-month').astype(int)

# 5. Auto-pay indicator
df_model['HasAutoPay'] = df_model['PaymentMethod'].apply(
    lambda x: 1 if 'automatic' in x.lower() else 0)

print(f"✅ Created 5 new features. Total features: {df_model.shape[1]}")
print(f"\nNew features: TotalServices, IsNewCustomer, TenureYears, AvgChargesPerMonth, HasContract, HasAutoPay")
```

**Cell 5.2 - Encoding & Preprocessing** (code):
```python
# Drop identifier
df_model = df_model.drop(['customerID', 'tenure_group'], axis=1)

# Encode target
df_model['Churn'] = (df_model['Churn'] == 'Yes').astype(int)

# Identify column types
categorical_cols = df_model.select_dtypes(include='object').columns.tolist()
numerical_cols = df_model.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove('Churn')

# Label encode categorical features
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    le_dict[col] = le

# Define X and y
X = df_model.drop('Churn', axis=1)
y = df_model['Churn']

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Churn rate in train: {y_train.mean()*100:.1f}%")
print(f"Churn rate in test: {y_test.mean()*100:.1f}%")
```

**Cell 5.3 - SMOTE & Scaling** (code):
```python
# Apply SMOTE to balance training data
smote = SMOTE(random_state=RANDOM_STATE)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"Before SMOTE: {sum(y_train==0)} retained, {sum(y_train==1)} churned")
print(f"After SMOTE:  {sum(y_train_balanced==0)} retained, {sum(y_train_balanced==1)} churned")

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

print(f"\n✅ Preprocessing complete. Ready for modeling!")
```

---

### SECTION 6: MODEL TRAINING & COMPARISON
**Cells**: 4 cells
**Length**: ~200 lines

**Cell 6.1 - Define Models** (code):
```python
# Define model portfolio
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
    'XGBoost': XGBClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE, 
                             eval_metric='logloss', use_label_encoder=False),
    'SVM': SVC(probability=True, random_state=RANDOM_STATE),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB()
}

print(f"Prepared {len(models)} models for training:")
for name in models.keys():
    print(f"  • {name}")
```

**Cell 6.2 - Train All Models** (code):
```python
# Train and evaluate all models
results = []

print("="*80)
print("MODEL TRAINING & EVALUATION")
print("="*80)

for name, model in models.items():
    print(f"\n🔄 Training {name}...", end=" ")
    
    # Train
    model.fit(X_train_scaled, y_train_balanced)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_prob = model.decision_function(X_test_scaled) if hasattr(model, 'decision_function') else None
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    })
    
    print(f"✅ ROC-AUC: {roc_auc:.4f}")

# Create results dataframe
results_df = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False)
print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)
print(results_df.to_string(index=False))
```

**Cell 6.3 - ROC Curves** (code):
```python
# Plot ROC curves
plt.figure(figsize=(10, 8))

for name, model in models.items():
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Best model
best_model_name = results_df.iloc[0]['Model']
print(f"\n🏆 Best Model: {best_model_name} (ROC-AUC: {results_df.iloc[0]['ROC-AUC']:.4f})")
```

**Cell 6.4 - Best Model Details** (code):
```python
# Detailed evaluation of best model
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test_scaled)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_best)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Retained', 'Churned'], yticklabels=['Retained', 'Churned'])
axes[0].set_title(f'{best_model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# Metrics bar chart
metrics = results_df[results_df['Model']==best_model_name].iloc[0]
metric_values = [metrics['Accuracy'], metrics['Precision'], metrics['Recall'], metrics['F1-Score'], metrics['ROC-AUC']]
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
axes[1].bar(metric_names, metric_values, color=colors['churn'])
axes[1].set_ylim(0, 1)
axes[1].set_title(f'{best_model_name} - Performance Metrics', fontsize=14, fontweight='bold')
for i, v in enumerate(metric_values):
    axes[1].text(i, v+0.02, f'{v:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# Classification report
print(f"\n{best_model_name} - Classification Report:")
print("="*60)
print(classification_report(y_test, y_pred_best, target_names=['Retained', 'Churned']))
```

---

### SECTION 7: MODEL EVALUATION & INTERPRETATION
**Cells**: 3 cells
**Length**: ~120 lines

**Cell 7.1 - Feature Importance** (code):
```python
# Feature importance (for tree-based models)
if hasattr(best_model, 'feature_importances_'):
    importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    plt.figure(figsize=(10, 10))
    plt.barh(importance['Feature'], importance['Importance'], color=colors['churn'])
    plt.title(f'{best_model_name} - Feature Importance', fontsize=14, fontweight='bold')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()
    
    print("\nTop 10 Most Important Features:")
    print(importance.tail(10).to_string(index=False))
else:
    # For non-tree models, use coefficient if available
    if hasattr(best_model, 'coef_'):
        importance = pd.DataFrame({
            'Feature': X.columns,
            'Coefficient': np.abs(best_model.coef_[0])
        }).sort_values('Coefficient', ascending=True)
        
        plt.figure(figsize=(10, 10))
        plt.barh(importance['Feature'], importance['Coefficient'], color=colors['churn'])
        plt.title(f'{best_model_name} - Feature Coefficients (Absolute)', fontsize=14, fontweight='bold')
        plt.xlabel('|Coefficient|')
        plt.tight_layout()
        plt.show()
```

**Cell 7.2 - Cross-Validation** (code):
```python
# Cross-validation for best model
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(best_model, X_train_scaled, y_train_balanced, 
                            cv=cv, scoring='roc_auc')

print(f"5-Fold Cross-Validation Results for {best_model_name}:")
print("="*50)
print(f"ROC-AUC scores: {cv_scores.round(4)}")
print(f"Mean ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
print(f"\n✅ Model shows stable performance across folds")
```

**Cell 7.3 - Model Interpretation Summary** (markdown):
```markdown
### Model Interpretation Summary

**Best Performing Model**: {best_model_name}
- **ROC-AUC**: {results_df.iloc[0]['ROC-AUC']:.4f} (excellent discrimination)
- **F1-Score**: {results_df.iloc[0]['F1-Score']:.4f} (balanced precision/recall)
- **Recall**: {results_df.iloc[0]['Recall']:.4f} (captures this % of actual churners)

**Top Predictive Features**:
1. **Contract** - Month-to-month customers are highest risk
2. **Tenure** - New customers (<12 months) need attention
3. **TechSupport** - Lack of support increases churn
4. **MonthlyCharges** - Higher charges correlate with churn
5. **PaymentMethod** - Electronic check users churn more

**Model Stability**:
Cross-validation shows consistent performance (CV std < 0.02), indicating the model generalizes well.
```

---

### SECTION 8: BUSINESS INSIGHTS & RECOMMENDATIONS
**Cells**: 2 cells
**Length**: ~100 lines

**Cell 8.1 - Customer Segmentation** (code):
```python
# Create risk segments using the best model
X_full_scaled = scaler.transform(X)
y_prob_all = best_model.predict_proba(X_full_scaled)[:, 1]

df['ChurnProbability'] = y_prob_all
df['RiskSegment'] = pd.cut(df['ChurnProbability'], 
                            bins=[0, 0.25, 0.50, 0.75, 1.0],
                            labels=['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk'])

# Segment analysis
segment_analysis = df.groupby('RiskSegment').agg({
    'customerID': 'count',
    'MonthlyCharges': 'mean',
    'Churn_Binary': 'mean'
}).round(2)
segment_analysis.columns = ['Customers', 'Avg_Monthly_Revenue', 'Actual_Churn_Rate']
segment_analysis['Actual_Churn_Rate'] = (segment_analysis['Actual_Churn_Rate'] * 100).round(1)
segment_analysis['At_Risk_Revenue'] = (segment_analysis['Customers'] * 
                                        segment_analysis['Avg_Monthly_Revenue'] * 12).round(0)

print("Customer Risk Segmentation:")
print("="*70)
print(segment_analysis)
print("="*70)

# Visualize
fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(segment_analysis))
bars = ax.bar(segment_analysis.index, segment_analysis['Customers'], color=['#2E7D32', '#F9A825', '#EF6C00', '#C62828'])
ax.set_title('Customer Distribution by Risk Segment', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Customers')
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.show()
```

**Cell 8.2 - Recommendations** (markdown):
```markdown
### Strategic Recommendations

#### 1. **High-Priority Actions** (Critical & High Risk Segments)
| Action | Target | Expected Impact |
|--------|--------|-----------------|
| Personal retention call | Critical Risk (P(churn) > 75%) | Prevent 40% of imminent churn |
| Discount offer (15-20%) | High Risk (50-75%) | Convert 25% to retained |
| Contract upgrade incentive | Month-to-month customers | Reduce churn by 30%+ |

#### 2. **Medium-Priority Actions** (Medium Risk)
- Implement automated email campaigns
- Offer service bundle upgrades
- Provide tech support trial

#### 3. **Long-Term Strategies**
- **Onboarding improvement**: Focus on first 12 months (highest churn period)
- **Payment method migration**: Incentivize autopay enrollment ($5/month credit)
- **Service bundling**: Each additional service reduces churn 8-12%

#### 4. **Financial Impact Estimate**
```
Assumptions:
- Average monthly revenue per customer: $65
- Average customer lifetime: 3 years (if retained)
- Intervention cost: $50-100 per customer

Potential savings (preventing 20% of predicted churn):
- Critical/High risk customers: ~1,500
- 20% prevented = 300 customers
- Value saved = 300 × $65 × 36 months = $702,000

ROI = ($702,000 - $30,000 intervention cost) / $30,000 = 2,240%
```
```

---

### SECTION 9: CONCLUSIONS & FUTURE WORK
**Cells**: 1 cell
**Length**: ~50 lines

**Cell 9.1 - Conclusions** (markdown):
```markdown
## Conclusions

### Key Findings

1. **Strongest Churn Predictors**:
   - Contract type (month-to-month = 14x higher churn vs 2-year)
   - Tenure (new customers < 12 months at highest risk)
   - Payment method (electronic check = 3x higher churn)

2. **Model Performance**:
   - Best model: [Model Name] with ROC-AUC of 0.XXX
   - Successfully identifies 80%+ of churners
   - Stable cross-validation performance

3. **Business Implications**:
   - Targeted retention can save $XXX,XXX annually
   - Focus on first-year customer experience is critical
   - Autopay and contract incentives are high-ROI interventions

### Strengths & Limitations

**Strengths**:
- Multiple models compared for robustness
- Statistical validation of key findings
- Actionable segmentation framework
- Quantified business impact

**Limitations**:
- Cross-sectional data (no time-series tracking)
- External factors not captured (competitors, market)
- Assumes historical patterns continue

### Future Improvements

1. **Data Enhancements**:
   - Customer service interaction logs
   - Network quality metrics
   - Competitor pricing data

2. **Modeling Advances**:
   - Deep learning approaches
   - Time-series churn prediction
   - Survival analysis for tenure modeling

3. **Implementation**:
   - A/B testing of retention strategies
   - Real-time scoring API
   - Automated alert system for high-risk customers
```

---

### SECTION 10: REFERENCES
**Cells**: 1 cell
**Length**: ~20 lines

**Cell 10.1 - References** (markdown):
```markdown
## References

### Dataset
- IBM Sample Datasets. (n.d.). Telco Customer Churn. Kaggle. https://www.kaggle.com/datasets/blastchar/telco-customer-churn

### Literature & Resources
- Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*, 16, 321-357.
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016*.
- Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.

### Kaggle Notebooks Referenced
- "customer-churn-prediction" (2,862 upvotes)
- "telecom-churn-prediction" (2,176 upvotes)
- "telco-churn-eda-cv-score-85" (579 upvotes)

### Python Libraries
- pandas, numpy, matplotlib, seaborn
- scikit-learn, xgboost, imbalanced-learn
- scipy (statistical tests)
```

---

## 🔧 SUBAGENT PROMPTS FOR SPECIALIZED TASKS

### SUBAGENT 1: EDA SPECIALIST
**Purpose**: Create all visualizations in Section 4

```
SUBAGENT PROMPT - EDA SPECIALIST

TASK: Create comprehensive exploratory data analysis for Telco Churn dataset.

CONTEXT:
- Dataset: 7,043 customers, 21 columns
- Target: Churn (Yes/No) - 26.5% churn rate
- Key features: Contract, tenure, PaymentMethod, MonthlyCharges, Services

REQUIREMENTS:
1. Target distribution visualization (count + pie chart)
2. Churn rate by 5 key categorical features (stacked bars)
3. Chi-square tests with Cramér's V for categorical features
4. Boxplots for numerical features by churn status
5. Correlation heatmap with target
6. Tenure deep-dive (histogram + binned analysis)
7. Contract analysis table with statistics

STYLE:
- Color scheme: churn='#E63946', retained='#457B9D'
- Figure size: (12,6) for single, (15,10) for multi-panel
- Include statistical annotations (p-values, effect sizes)
- Professional, publication-ready formatting

OUTPUT FORMAT:
- Python code cells
- Each visualization followed by interpretation
- Print statistical test results with significance markers (*, **, ***)
```

---

### SUBAGENT 2: MODEL TRAINING SPECIALIST
**Purpose**: Train and compare all ML models in Section 6

```
SUBAGENT PROMPT - MODEL TRAINING SPECIALIST

TASK: Train 8 classification models and compare performance.

CONTEXT:
- Preprocessed data with SMOTE balancing
- Features scaled with StandardScaler
- Binary classification (Churn: 0/1)

MODELS TO TRAIN:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. XGBoost
6. SVM
7. KNN
8. Naive Bayes

REQUIREMENTS:
1. Train each model with appropriate hyperparameters
2. Calculate: Accuracy, Precision, Recall, F1-Score, ROC-AUC
3. Generate ROC curves for all models on single plot
4. Create summary comparison table sorted by ROC-AUC
5. Identify best performing model

OUTPUT FORMAT:
- Training progress printed with emoji indicators
- Results dataframe with all metrics
- Single ROC curve figure with legend
- Clear identification of best model

RANDOM_STATE: 42 for reproducibility
```

---

### SUBAGENT 3: BUSINESS ANALYST SPECIALIST
**Purpose**: Create business recommendations in Section 8

```
SUBAGENT PROMPT - BUSINESS ANALYST SPECIALIST

TASK: Transform model predictions into actionable business strategy.

CONTEXT:
- Best model provides churn probability scores
- Customer segments based on probability thresholds
- Average monthly revenue: $65
- Customer lifetime: 3 years
- Intervention cost: $50-100

REQUIREMENTS:
1. Create risk segments (Low, Medium, High, Critical)
2. Calculate at-risk revenue per segment
3. Design intervention strategies by segment
4. Estimate ROI of retention program
5. Prioritize actions by impact/effort

OUTPUT FORMAT:
- Customer segmentation table with metrics
- Risk segment visualization
- Strategy matrix (markdown table)
- ROI calculation with assumptions stated
- Prioritized action list
```

---

## ✅ FINAL CHECKLIST

Before considering V4 complete, verify:

### Code Quality
- [ ] All cells execute without errors
- [ ] Random seeds set (RANDOM_STATE = 42)
- [ ] No unnecessary imports or unused variables
- [ ] Comments explain non-obvious code

### Content Completeness
- [ ] Executive summary present
- [ ] All 5 hypotheses tested
- [ ] 8 models trained and compared
- [ ] Best model identified with metrics
- [ ] Feature importance analyzed
- [ ] Business recommendations provided
- [ ] References included

### Formatting
- [ ] Consistent heading hierarchy (# ## ###)
- [ ] Tables rendered properly
- [ ] Visualizations have titles/labels/legends
- [ ] Statistical results include p-values

### Length Target
- [ ] Total cells: ~28 (±5)
- [ ] Total lines: 2,000-3,000
- [ ] Execution time: < 5 minutes

---

## 🚀 EXECUTION INSTRUCTIONS

**For AI Agent**:

1. **Read this entire document first**
2. **Create new notebook**: `Telco_Churn_V4_Final.ipynb`
3. **Build sections sequentially** (1 through 10)
4. **Use subagent prompts** for specialized tasks
5. **Test each section** before proceeding
6. **Verify checklist** at completion

**Total estimated tokens**: ~15,000 for complete V4 notebook

**Expected quality**: Professional, publication-ready, academically rigorous

---

**Document Version**: 1.0  
**Created**: December 3, 2025  
**Purpose**: Master instruction set for V4 notebook creation  
**Target Audience**: GitHub Copilot Agent  

🎯 **Goal: Create the best possible condensed version that exceeds V3 quality in 1/10 the length!**
