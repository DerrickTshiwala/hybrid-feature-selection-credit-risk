# Improving Consumer Credit Risk Classification Using Hybrid Feature Selection and Gradient Boosting Models

This repository contains the core Python implementation and empirical framework for the research paper published in the **Proceedings of the 10th International Conference on Interdisciplinary Research on Computer Science, Psychology, and Education (ICICPE 2026), Vol. 10, No. 1, p. 408 (ISSN: 2508-7436)**.

## 📌 Project Overview
Traditional credit scoring models struggle with high-dimensional financial data, structural noise, and severe class imbalance. This project introduces a **two-tier hybrid feature selection methodology** coupled with **optimized Gradient Boosting Machine (GBM) architectures** to maximize consumer default predictive accuracy while minimizing portfolio risk exposure.

## 🛠️ Technical Architecture
The pipeline executes sequentially across three core layers:
1. **Data Engineering & Imbalance Handling:** Outlier extraction, feature standardization, and stratification adjustments.
2. **Hybrid Feature Selection Framework:**
   * *Filter Phase:* Low-variance drop followed by non-linear tracking via **Mutual Information (MI)** ranking.
   * *Wrapper Phase:* **Recursive Feature Elimination (RFE)** powered by a Random Forest base estimator to isolate pure predictive value and eradicate multicollinearity.
3. **Ensemble Modeling:** Comparative hyperparameter optimization between **XGBoost** and **LightGBM** architectures to achieve optimal threshold calibration.

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy pandas scikit-learn xgboost lightgbm matplotlib
```

### Execution
Run the primary modeling script to generate feature importance profiles, optimization thresholds, and classification metrics:
```bash
python hybrid_credit_risk.py
```

## 📊 Core Performance Metrics
* **Feature Dimensionality Reduction:** Effectively compresses noisy data metrics down to optimal predictive inputs without loss of variance.
* **Target Metric Focus:** Optimizes **AUROC** and minimizes False Negatives to safeguard loan portfolios against systemic defaults.

## 📖 Citation
If you utilize this framework or build upon this methodology in a professional/academic environment, please cite the canonical paper:
```text
Mubenga, D. T. (2026). "Improving Consumer Credit Risk Classification Using Hybrid Feature Selection and Gradient Boosting Models." Proceedings of the 10th ICICPE 2026, Vol. 10, No. 1, p. 408. ISSN: 2508-7436.
```
