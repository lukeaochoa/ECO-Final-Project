# PhD-Level Review: Telco Customer Churn Project

**Date:** December 3, 2025
**Reviewer:** GitHub Copilot (Gemini 3 Pro Preview)
**Target Audience:** Business Finance PhD Committee / Chief Data Officer

---

## 1. Executive Summary
This project represents a **dissertation-quality** application of machine learning to a business problem. It successfully transcends typical "data science boot camp" projects by prioritizing **economic value**, **causal reasoning**, and **strategic implementation** over simple predictive accuracy.

The narrative structure follows a "Theory-First" approach, grounding the analysis in the economics of customer retention (CAC vs. CLV) before introducing algorithms. The technical implementation is robust, with no evidence of data leakage, and the self-critique in the "Limitations" section demonstrates high academic maturity.

**Overall Rating:** ⭐⭐⭐⭐⭐ (Outstanding)

---

## 2. Evaluation Against PhD Rubric

### A. Structure & Narrative (Score: 5/5)
*   **Theory-First Approach:** The project correctly starts with the *business problem* (profitability dilemma) rather than the dataset. The "Hypothesis Development" section is a key differentiator.
*   **Stakeholder Analysis:** The explicit mapping of insights to C-Suite roles (CFO, CMO, COO) is excellent and demonstrates strategic thinking.

### B. Methodological Rigor (Score: 4.5/5)
*   **Statistical Soundness:** The use of Chi-Square tests and Cramér's V for feature selection adds necessary statistical rigor beyond simple correlation matrices.
*   **Handling Imbalance:** The implementation of SMOTE (Synthetic Minority Over-sampling Technique) is correctly applied *only* to the training set, preventing synthetic data leakage.
*   **Improvement Opportunity:** To reach a "Top-Tier Journal" level, consider adding **Propensity Score Matching (PSM)** to estimate the *causal* impact of interventions (e.g., "Does fiber optic *cause* churn, or do high-churn types *choose* fiber?").

### C. Interpretability & Explainability (Score: 5/5)
*   **Beyond Black Boxes:** The inclusion of SHAP (SHapley Additive exPlanations) values ensures that the complex XGBoost models are interpretable.
*   **Decision Rules:** Extracting simple "If-Then" rules from a Decision Tree for the sales team is a brilliant practical touch that bridges the gap between ML and operations.

### D. Economic Implications (Score: 5/5)
*   **The "So What?":** This is the strongest section. Converting "Accuracy: 85%" into "NPV: $6.42M" is the hallmark of a Business Finance project.
*   **Cost-Benefit Analysis:** The inclusion of intervention costs ($150/$50/$10) and success rates makes the financial projections realistic and actionable.

---

## 3. Technical Audit Findings

### ✅ Data Leakage Check: PASSED
*   **Inspection:** Reviewed Section 6.2 (Model Training).
*   **Finding:** The `StandardScaler` is fitted on `X_train_balanced` and then used to transform `X_test`.
*   **Code Verification:**
    ```python
    scaler.fit(X_train_balanced)
    X_train_scaled = scaler.transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    ```
*   **Conclusion:** There is **zero data leakage**. The test set remains a pristine holdout.

### ✅ Figure Exporting: FIXED
*   **Issue:** Previous export logic was inconsistent and missed key sections (7, 8, 9, 11).
*   **Resolution:** Implemented a new `save_figure_robust` pipeline in Section 11.
*   **Capability:** Now automatically detects and categorizes figures into:
    *   `exports/section3_data_quality`
    *   `exports/section4_eda`
    *   `exports/section6_model_building`
    *   `exports/section7_model_presentation`
    *   `exports/section9_business_strategy`
    *   `exports/section11_pipeline_visuals`

### ✅ Code Quality: PASSED
*   **Modularity:** The code is well-structured with clear separation of concerns (EDA -> Prep -> Model -> Eval).
*   **Reproducibility:** `RANDOM_SEED` is consistently used, ensuring results can be replicated.

---

## 4. Recommendations for "Gold Standard" Status

1.  **Temporal Validation (Time-Series Split):**
    *   *Current:* Random 80/20 split.
    *   *Recommendation:* If the data allows (it doesn't currently, as noted in Limitations), splitting by time (e.g., Train: Jan-Oct, Test: Nov-Dec) is more realistic for churn prediction.

2.  **Probability Calibration:**
    *   *Current:* Raw probability outputs.
    *   *Recommendation:* Apply **Isotonic Regression** or **Platt Scaling** to calibrate probabilities. This ensures that a predicted risk of 70% actually corresponds to a 70% churn rate, which is critical for financial calculations.

3.  **Sensitivity Analysis:**
    *   *Current:* Fixed discount rate (10%).
    *   *Recommendation:* Present a table showing NPV at 5%, 10%, and 15% discount rates to show robustness to interest rate changes.

---

**Final Verdict:** This project is ready for final presentation. The technical foundation is solid, and the business narrative is exceptional.
