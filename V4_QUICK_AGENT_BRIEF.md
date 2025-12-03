# 🎯 V4 PROJECT - QUICK AGENT BRIEF

## ONE-MINUTE SUMMARY

**Mission:** Create condensed, simplified, high-quality V4 notebook  
**Source:** V1-V3 notebooks + rubric (Group_project_2.docx) + 5 Kaggle examples  
**Target:** 60-80 cells (down from 185+), <10 min execution  
**Performance:** ROC-AUC ≥ 0.85, F1 ≥ 0.80, $5.2M business value  

---

## CRITICAL REQUIREMENTS

### ✅ MUST DO:
1. **Split every cell** (computation separate from visualization)
2. **Store all figures** in figures_dict
3. **Statistical rigor** (chi-square, t-tests, effect sizes)
4. **SMOTE on training only** (prevent data leakage)
5. **Cell splitting protocol** (EVERY mixed cell split)

### ❌ MUST AVOID:
1. Data leakage (scale before split)
2. Mixed computation/visualization cells
3. Missing figures_dict storage
4. Accuracy metric for imbalanced data
5. SMOTE on test set

---

## NOTEBOOK STRUCTURE (50-70 cells)

```
Section 0: Introduction (5-7 cells)
Section 1: Data Foundation (4-6 cells)
  - Imports, loading, quality, cleaning
  - Encoding, train-test split, SMOTE, scaling
Section 2: EDA (12-15 cells)
  - Target analysis (2)
  - Top 4 categorical features (8)
  - Top 3 numerical features (6)
  - Correlation (2)
Section 3: Models (8-12 cells)
  - 5-6 models (down from 11)
  - Cross-validation, tuning, comparison
Section 4: Interpretation (6-8 cells)
  - Feature importance, SHAP, decision rules
Section 5: Business Strategy (6-8 cells)
  - ML insights → operational actions
  - Theory-based strategic recommendations
Section 6: Conclusions (3-5 cells)
```

---

## KEY SIMPLIFICATIONS FROM V3

| Aspect | V3 | V4 | Why |
|--------|-----|-----|-----|
| Total Cells | 185+ | 50-70 | Remove redundancy |
| Models | 11 | 5-6 | Focus on top performers |
| EDA Features | All 21 | Top 7 | High-impact only |
| Engineered Features | 12 | 0 | NO FEATURE ENGINEERING |
| Business Strategy | Complex ROI | Simplified | ML insights → operations |
| Execution Time | 20+ min | <10 min | Faster grids/sampling |

---

## TOP 7 FEATURES TO ANALYZE (EDA)

**Categorical (4):**
1. Contract (strongest - 15x difference)
2. InternetService (fiber optic issues)
3. PaymentMethod (electronic check risk)
4. TechSupport (26.5% protective effect)

**Numerical (3):**
1. Tenure (lifecycle - first 12 months critical)
2. MonthlyCharges (price sensitivity threshold)
3. TotalCharges (cumulative value)

---

## NO FEATURE ENGINEERING

**Use original 21 features only:**
- Data cleaning and imputation only
- Categorical encoding (one-hot, label encoding)
- Standard scaling for numerical features
- Focus on interpreting core business features

---

## 5-6 CORE MODELS

1. **Logistic Regression** (interpretable baseline)
2. **Random Forest** (ensemble power)
3. **XGBoost** (gradient boosting)
4. **LightGBM** (best from V3: 0.8367 ROC-AUC)
5. **Stacking Ensemble** (meta-learner)
6. SVM (optional if time permits)

---

## VISUALIZATION STANDARDS

```python
# Colors
COLORS = {
    'churn': '#e74c3c',      # Red
    'retained': '#2ecc71',   # Green
    'high_risk': '#e74c3c',
    'medium_risk': '#f39c12',
    'low_risk': '#2ecc71'
}

# Sizes
FIGURE_SIZES = {
    'small': (10, 6),
    'medium': (15, 8),
    'large': (20, 12),
    'wide': (18, 6)
}

# Every plot MUST have:
- Clear title (14-16pt, bold)
- Axis labels (11-12pt)
- Statistical annotations (p-values, effect sizes)
- Business insight box
- Stored in figures_dict
- 300 DPI minimum
```

---

## CELL SPLITTING TEMPLATE

```python
# ===== COMPUTATION CELL =====
# Purpose: [What this does]
# Step 1: Calculate statistics
result = calculate_something()

# Step 2: Statistical test
chi2, p_value = chi2_contingency(...)
effect_size = calculate_effect_size()

print(f"✅ Complete: {metric}")
```

```python
# ===== VISUALIZATION CELL =====
fig, ax = plt.subplots(figsize=FIGURE_SIZES['medium'])

# Plot with annotations
# ... plotting code ...

# Statistical annotation
ax.text(x, y, f'p < 0.001, Effect: {size}')

# Business insight
ax.text(x, y, 'Key Insight: [business impact]')

figures_dict['section_X_description'] = fig
plt.show()
```

---

## EXECUTION WORKFLOW

### PHASE 1: Setup (Parallel)
```python
# Read all these simultaneously:
- Group_project_2.docx (rubric)
- PROJECT_SPECIFICATION_V3.md (blueprint)
- COMPREHENSIVE_PROJECT_CONTEXT_FOR_AI_HANDOFF.md (history)
- V3 notebook structure
- Top Kaggle examples
```

### PHASE 2: Plan
```python
manage_todo_list(operation='write', todoList=[
    {id: 1, title: 'Extract rubric requirements', status: 'not-started'},
    {id: 2, title: 'Create notebook skeleton', status: 'not-started'},
    {id: 3, title: 'Implement Section 0', status: 'not-started'},
    # ... etc (17 total tasks)
])
```

### PHASE 3: Execute
For each section:
1. Mark todo in-progress
2. Create cells (markdown → computation → visualization)
3. Execute and validate
4. Mark todo completed
5. Save progress

### PHASE 4: Validate
- Run All (must complete in <10 min)
- Check metrics (ROC-AUC ≥ 0.85)
- Verify figures_dict (20+ plots)
- get_errors (must be clean)

---

## SUBAGENT TASKS

Deploy in parallel:

**Batch 1 (Context):**
1. Extract rubric from Group_project_2.docx
2. Optimize import structure
3. Create EDA template

**Batch 2 (Technical):**
4. Validate feature engineering (top 8)
5. Optimal model selection (top 5)
6. SHAP best practices

**Batch 3 (Business):**
7. Financial model validation (industry benchmarks)

---

## QUALITY CHECKLIST

### Code Quality:
- [ ] All cells split (computation vs visualization)
- [ ] All functions have docstrings
- [ ] Validation after transformations
- [ ] Random seed = 42 everywhere
- [ ] No TODOs left in code

### Data Quality:
- [ ] No data leakage (scale after split)
- [ ] SMOTE on training only
- [ ] Stratified train-test split
- [ ] Cross-validation implemented
- [ ] Proper metrics (F1, ROC-AUC, not accuracy)

### Visual Quality:
- [ ] All plots in figures_dict
- [ ] 300 DPI minimum
- [ ] Consistent color scheme
- [ ] Statistical annotations
- [ ] Business insight boxes

### Business Quality:
- [ ] ROI calculations for strategies
- [ ] NPV/IRR for financial modeling
- [ ] Prioritized recommendations
- [ ] Implementation roadmap
- [ ] Risk assessment

---

## KEY NUMBERS TO ACHIEVE

**Model Performance:**
- ROC-AUC: ≥ 0.85 (V3: 0.8367)
- F1-Score: ≥ 0.80 (V3: 0.6258)
- Recall: ≥ 0.75 (catch churners)
- Precision: ≥ 0.60 (minimize false alarms)

**Business Value:**
- Net savings: $5.2M annually
- ROI: 350%
- Payback: 3-4 months
- 6 strategic recommendations

**Project Metrics:**
- Total cells: 60-80
- Execution time: <10 minutes
- Visualizations: 20+ professional plots
- Features analyzed: Top 7 (not all 21)

---

## COMMON ERRORS TO PREVENT

1. **Data Leakage:**
```python
# ❌ WRONG
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled)

# ✅ CORRECT
X_train, X_test = train_test_split(X)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

2. **SMOTE Misuse:**
```python
# ❌ WRONG
X_train, X_test = train_test_split(X, y)
X_resampled, y_resampled = SMOTE().fit_resample(X_train + X_test)

# ✅ CORRECT
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
X_train_smote, y_train_smote = SMOTE().fit_resample(X_train, y_train)
```

3. **Cell Mixing:**
```python
# ❌ WRONG (computation + viz in one cell)
result = calculate()
plt.plot(result)
plt.show()

# ✅ CORRECT (split into 2 cells)
# Cell 1: result = calculate()
# Cell 2: plt.plot(result); plt.show()
```

---

## TOOLS TO USE

**Mandatory:**
- `manage_todo_list` (track progress)
- `copilot_getNotebookSummary` (before editing)
- `edit_notebook_file` (add cells)
- `run_notebook_cell` (validate)
- `get_errors` (check quality)

**Frequently:**
- `read_file` (parallel for context)
- `semantic_search` (find examples)
- `grep_search` (exact patterns)
- `runSubagent` (complex tasks)

**As Needed:**
- `microsoft_docs_search` (ML libraries)
- `vscode_websearchforcopilot_webSearch` (latest practices)
- `multi_replace_string_in_file` (batch edits)

---

## SUCCESS INDICATORS

**You'll know you're on track when:**
- ✅ Todo list progressing steadily
- ✅ Each section takes 30-60 minutes
- ✅ Cells execute without errors immediately
- ✅ Visualizations look professional
- ✅ Statistical tests confirm significance
- ✅ Business insights are clear
- ✅ Code is clean and documented

**Red flags to address:**
- ⚠️ Execution errors piling up
- ⚠️ Cells taking >5 minutes to run
- ⚠️ Visualizations without figures_dict
- ⚠️ Missing statistical tests
- ⚠️ No business context for findings
- ⚠️ Hyperparameter tuning >5 minutes

---

## FINAL DELIVERABLES

**Primary:**
- Telco Project v4.ipynb (60-80 cells, error-free)

**Supporting:**
- V4_NOTEBOOK_SUMMARY.md (executive overview)
- V4_EXECUTION_REPORT.md (metrics, timing)
- V4_FIGURES_EXPORT.py (export script)
- V4_RUBRIC_COMPLIANCE.md (checklist)
- PRESENTATION_GUIDE_V4.md (slide structure)

---

## 🚀 READY TO START!

**First 3 actions:**
1. Read Group_project_2.docx (rubric requirements)
2. Create todo list with 17 tasks
3. Deploy subagents 1-3 for context analysis

**Remember:**
- V4 is CONDENSED, SIMPLIFIED, HIGH QUALITY
- Follow cell splitting protocol always
- Use parallel operations whenever possible
- Validate early and often
- Business focus in every section

**GO BUILD V4! 💪**

---

**For full instructions, see: V4_COMPREHENSIVE_INSTRUCTION_PROMPT.md**
