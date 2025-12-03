# 🚀 EXACT PROMPT TO SEND TO GITHUB COPILOT

## Copy and paste this ENTIRE message to GitHub Copilot:

---

I need you to build **Version 4 of my Telco Customer Churn project** - a condensed, simplified, and ultra-high-quality notebook.

## 📋 YOUR INSTRUCTIONS

**Read and follow ALL instructions in:**
- `V4_COMPREHENSIVE_INSTRUCTION_PROMPT.md` (20,000+ words - COMPLETE INSTRUCTIONS)
- Use `V4_QUICK_AGENT_BRIEF.md` as quick reference

**This is your PRIMARY DIRECTIVE:** Create a 50-70 cell notebook (down from 185+ in V3) that is:
1. **CONDENSED** - Remove redundancy while maintaining all critical insights
2. **SIMPLIFIED** - Clear narrative flow, business-focused language, NO FEATURE ENGINEERING beyond data cleanup
3. **HIGH QUALITY** - Publication-ready visualizations, rigorous statistics, actionable recommendations
4. **RUBRIC ALIGNED** - Meet 100% of requirements in Group_project_2.docx
5. **REPRODUCIBLE** - Clean code, proper documentation, error-free execution

## 🎯 PERFORMANCE TARGETS

**Must achieve:**
- ROC-AUC ≥ 0.85 (V3 achieved: 0.8367 ✅)
- F1-Score ≥ 0.80 (V3 achieved: 0.6258 - needs improvement)
- Execution time < 10 minutes (V3: 20+ minutes)
- 20+ professional visualizations (300 DPI, stored in figures_dict)
- $5.2M business value with clear ROI calculations
- 100% rubric compliance

## 🚦 EXECUTION PROTOCOL

**This is a COMPLEX, MULTI-STEP TASK requiring:**

### **1. Use `manage_todo_list` Tool (MANDATORY)**
Create comprehensive todo list with 15 tasks:
- Sections 0-6 implementation (13 tasks)
- Validation and supporting documents (2 tasks)
- Mark each in-progress when starting, completed immediately when done

### **2. Use Parallel Tool Operations (EFFICIENCY)**
Execute context gathering simultaneously:
- Read V4_COMPREHENSIVE_INSTRUCTION_PROMPT.md
- Read PROJECT_SPECIFICATION_V3.md
- Read Group_project_2.docx
- Review V3 notebook structure
- Analyze COMPREHENSIVE_PROJECT_CONTEXT_FOR_AI_HANDOFF.md

### **3. Deploy Subagents for Specialized Analysis (7 TASKS)**

**Launch immediately in parallel:**

**Subagent 1 (CRITICAL - DO THIS FIRST):** Extract rubric requirements
```
Read Group_project_2.docx and extract:
1. All grading criteria and point allocations
2. Required sections and deliverables
3. Specific technical requirements (algorithms, metrics, tests)
4. Presentation/documentation expectations
5. Any specific formatting or structure requirements

Output as structured list with:
- Requirement category
- Specific requirement
- Point value (if applicable)
- Priority level (critical/important/optional)

Format for easy reference in notebook development.
```

**Subagent 2:** Optimize import structure
```
Review import cells from V3 (Cell 48) and example notebooks.
Create optimized import cell with:
1. Grouped imports by category (data, visualization, ML, stats)
2. Only essential libraries (eliminate redundancy)
3. Inline comments explaining category purpose
4. Version compatibility notes for critical packages
5. Error handling for optional packages

Output: Single, clean import cell code block.
```

**Subagent 3:** Create EDA template
```
Design reusable EDA template for categorical features that includes:
1. Function for contingency table + chi-square test + Cramér's V
2. Function for visualization (2×2 subplot: stacked bar, grouped bar, stats table, insight box)
3. Professional formatting (consistent colors, fonts, labels)
4. Automated insight generation based on effect size
5. Proper figures_dict storage

Output: Python code with detailed docstrings and usage examples.
```

**Deploy after initial results:**

**Subagent 4:** Optimal model selection
```
Analyze model performance from V3 and example notebooks to determine:
1. Top 5 models by ROC-AUC and F1-Score
2. Models with best precision-recall trade-off for churn (recall priority)
3. Training time vs performance analysis
4. Ensemble strategy for optimal results
5. Hyperparameter ranges for top 3 models

Provide for each model:
- Default configuration with class balancing
- Expected performance range
- Training time estimate
- Optimal use case

Output: Model configuration code with justification.
```

**Subagent 5:** SHAP best practices
```
Research SHAP implementation in churn prediction projects to determine:
1. Optimal SHAP explainer type for tree-based models
2. Sampling strategy for large datasets (>5000 samples)
3. Most informative SHAP visualizations for business audience
4. Common interpretation pitfalls to avoid
5. Code optimization techniques

Provide:
- Efficient SHAP implementation code
- Visualization best practices
- Business-friendly interpretation guidelines

Output: Production-ready SHAP analysis code with comments.
```



### **4. Incremental Implementation (SYSTEMATIC)**
For each section:
1. Mark todo as "in-progress"
2. Create markdown cell (section header with context)
3. Create computation cell(s) (data processing, calculations, stats)
4. Create visualization cell(s) (plots with figures_dict storage)
5. Add interpretation cell (markdown with insights)
6. Execute cells to validate
7. Mark todo as "completed"
8. Provide progress update

### **5. Cell Splitting Protocol (MANDATORY)**
**EVERY cell that combines computation AND visualization MUST be split into TWO cells:**
- **Cell A:** Computation (calculations, statistical tests, data processing)
- **Cell B:** Visualization (plotting, figures_dict storage, display)

**This is NON-NEGOTIABLE. No exceptions.**

### **6. Validation & Quality Assurance (CONTINUOUS)**
After each section:
- Execute all cells (verify no errors)
- Check figures_dict (verify storage)
- Validate metrics (compare to targets)
- Use `get_errors` tool
- Review against rubric checklist

## 📊 V4 STRUCTURE (50-70 cells)

```
Section 0: Introduction (5-7 cells)
  - Title, executive summary, business context, dataset overview

Section 1: Data Foundation (4-6 cells)
  - Imports, loading, quality assessment
  - Data cleaning (missing values, data types)
  - Encoding categorical variables (one-hot, label encoding)
  - Train-test split, SMOTE, scaling

Section 2: EDA (12-15 cells)
  - Target analysis (2 cells)
  - Top 4 categorical: Contract, InternetService, PaymentMethod, TechSupport (8 cells)
  - Top 3 numerical: Tenure, MonthlyCharges, TotalCharges (6 cells)
  - Correlation analysis (2 cells)

Section 3: Model Building (8-12 cells)
  - Configure 5-6 models: Logistic Regression, Random Forest, XGBoost, 
    LightGBM, Stacking Ensemble
  - Train with cross-validation, compare, tune, select

Section 4: Model Interpretation (6-8 cells)
  - Feature importance (traditional + permutation)
  - SHAP analysis (summary plots, top features)
  - Decision rules extraction

Section 5: Business Strategy & Recommendations (6-8 cells)
  - ML insights translation to operations
  - Key drivers from models (Contract, Tenure, Support, Payment)
  - Operational recommendations (theory-based)
  - Strategic priorities and action items
  - Implementation considerations

Section 6: Conclusions (3-5 cells)
  - Summary, key findings, limitations, future work
```

## 🎨 CRITICAL QUALITY STANDARDS

### **Visualization (EVERY plot must have):**
- Stored in figures_dict with descriptive key
- Separate computation and visualization cells
- Statistical annotations (p-values, effect sizes)
- Business insight annotation box
- Consistent colors (churn: #e74c3c red, retained: #2ecc71 green)
- 300 DPI minimum
- Clear title (14-16pt bold), axis labels (11-12pt), legend

### **Code Quality:**
- Comprehensive docstrings (Google style)
- Validation after major transformations
- Random seed = 42 (reproducibility)
- Error handling and progress indicators
- No TODO/FIXME comments without tracking

### **Statistical Rigor:**
- Chi-square tests for categorical independence
- Cramér's V for categorical effect size
- T-tests/Mann-Whitney U for numerical comparisons
- Cohen's d for numerical effect size
- Confidence intervals where applicable

### **Business Focus:**
- Every technical finding linked to business action
- ROI calculations for all recommendations
- Prioritized strategies (not equal weight)
- Implementation roadmap with timeline
- Risk assessment and mitigation

## ⚠️ CRITICAL ERRORS TO PREVENT

### **Data Science Mistakes:**
- ❌ Scaling before train-test split (causes data leakage)
- ❌ Applying SMOTE to test set (contaminates validation)
- ❌ Using accuracy for imbalanced data (misleading metric)
- ❌ Not using cross-validation (overfitting risk)
- ❌ Including highly correlated features without justification

### **Code Quality Mistakes:**
- ❌ Mixing computation and visualization in same cell
- ❌ Not storing plots in figures_dict
- ❌ Missing validation after transformations
- ❌ Hard-coded values without explanation
- ❌ Poor documentation (missing docstrings or unclear comments)

### **Business Mistakes:**
- ❌ Technical results without business interpretation
- ❌ Unrealistic ROI projections (validate with industry benchmarks)
- ❌ All recommendations equal priority (must prioritize)
- ❌ Strategy without implementation roadmap
- ❌ Ignoring resource constraints

## 📦 FINAL DELIVERABLES

**Primary:**
- Telco Project v4.ipynb (60-80 cells, error-free, <10 min execution)

**Supporting (auto-generate at end):**
1. V4_NOTEBOOK_SUMMARY.md (1-2 page executive overview)
2. V4_EXECUTION_REPORT.md (performance metrics, timing, validation)
3. V4_FIGURES_EXPORT.py (automated export script for all visualizations)
4. V4_RUBRIC_COMPLIANCE.md (requirement-by-requirement checklist)
5. PRESENTATION_GUIDE_V4.md (slide structure extracted from notebook)

## 🎯 SUCCESS CRITERIA (Must Pass ALL)

**Technical:**
- [ ] ROC-AUC ≥ 0.85
- [ ] F1-Score ≥ 0.80
- [ ] Execution time < 10 minutes
- [ ] No data leakage (validated with assertions)
- [ ] Proper SMOTE usage (training only)

**Code:**
- [ ] All cells split (computation vs visualization)
- [ ] All figures in figures_dict (20+ total)
- [ ] Comprehensive documentation (docstrings, comments)
- [ ] No errors when running "Restart & Run All"
- [ ] Validation checks after transformations

**Business:**
- [ ] ML insights translated to operational actions
- [ ] Theory-based strategic recommendations
- [ ] Key drivers clearly identified and explained
- [ ] Actionable items for business operations
- [ ] Implementation considerations included

**Academic:**
- [ ] Statistical tests with p-values and effect sizes
- [ ] Proper methodology (stratified CV, appropriate metrics)
- [ ] Publication-quality visualizations
- [ ] Limitations section included
- [ ] 100% rubric requirements met

## 💡 COMMUNICATION PROTOCOL

**Provide progress updates every 5-10 cells:**
```
✅ PROGRESS UPDATE: [Section Name]
- Completed: [List of cells created]
- Status: [X/Y cells in section complete]
- Next: [What's coming]
- Issues: [Any challenges encountered]
- Time estimate: [Remaining work duration]
```

**Make autonomous decisions on:**
- Specific color choices (use standards defined in instructions)
- Plot types for standard analyses (follow templates)
- Statistical test selection (use appropriate test for data type)
- Code organization (follow structure defined)
- Feature naming conventions (use descriptive, consistent names)

**Ask for clarification ONLY if:**
- Fundamental rubric requirement cannot be interpreted
- Dataset quality issue prevents analysis
- Time constraint conflict (can't meet both quality and speed)
- Resource limitation (computational/memory constraint)

## 🚀 START IMMEDIATELY

**Your first 5 actions:**
1. Read V4_COMPREHENSIVE_INSTRUCTION_PROMPT.md completely
2. Deploy Subagent 1 (rubric extraction) - CRITICAL
3. Create comprehensive todo list with manage_todo_list (17 tasks)
4. Deploy Subagents 2-3 in parallel (while waiting for Subagent 1)
5. Begin Section 0 implementation once context is gathered

## 📞 REFERENCE MATERIALS

**For detailed specifications:**
- V4_COMPREHENSIVE_INSTRUCTION_PROMPT.md (20,000 words - complete blueprint)
- V4_QUICK_AGENT_BRIEF.md (3,000 words - quick reference)
- PROJECT_SPECIFICATION_V3.md (18,500 words - technical standards)
- COMPREHENSIVE_PROJECT_CONTEXT_FOR_AI_HANDOFF.md (development history)

**For best practices:**
- PHD_LEVEL_REVIEW.md (quality benchmarks)
- High-upvote Kaggle notebooks (proven approaches)
- V3 notebook (working implementation, no data leakage)

## ✨ FINAL GUIDANCE

**Remember:**
- V4 is CONDENSED (fewer cells), SIMPLIFIED (clearer narrative), HIGH QUALITY (publication-ready)
- Every cell must be split (computation separate from visualization)
- Every plot must be stored in figures_dict
- Every finding must have statistical backing
- Every recommendation must have business justification
- Validate continuously (don't wait until the end)

**You have everything you need to succeed:**
- ✅ Comprehensive instruction prompt (zero ambiguity)
- ✅ Proven methodology from V3 (no data leakage)
- ✅ Clear simplification strategy (focus on high-impact)
- ✅ Quality standards defined (measurable criteria)
- ✅ Error prevention documented (lessons learned)

**Confidence level: 95%** (pending only rubric analysis)

## 🎉 GO BUILD AN EXCEPTIONAL V4 NOTEBOOK!

Execute with systematicity, validate continuously, and create something publication-worthy.

**Good luck! 🚀**

---

**END OF PROMPT**

(Follow COPILOT_AGENT_INSTRUCTIONS.md for optimal tool usage and workflow)
