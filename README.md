# IPL Delivery-Level Performance Analytics
## Predicting Match Outcomes Using Ball-by-Ball Data

**Course Assignment 1 | Supervised Machine Learning**  
**Supervisor:** Prof. Raja Hashim Ali

---

## Project Overview

This project applies supervised machine learning to IPL (Indian Premier League) ball-by-ball delivery data to predict delivery-level events — specifically whether a wicket falls, how many runs are scored, and what type of dismissal occurs. The study addresses seven research questions covering classification, regression, class imbalance, phase analysis, and feature engineering.

---

## Dataset

| Attribute | Details |
|-----------|---------|
| **Name** | IPL Deliveries Dataset |
| **Source** | [Kaggle — IPL Ball-by-Ball Data](https://www.kaggle.com/datasets/ramjidoolla/ipl-data-set) |
| **Rows** | 20,000 deliveries |
| **Columns** | 18 features |
| **Coverage** | 172 IPL matches, 8 teams |
| **Target (primary)** | `is_wicket` (binary classification) |
| **Target (secondary)** | `total_runs` (regression), `dismissal_type` (multi-class) |

---

## Repository Structure

```
ipl-ml-assignment/
│
├── RQ1_wicket_prediction.ipynb        # Wicket classification: Logistic Regression, RF, XGBoost, SVM
├── RQ2_bowler_analysis.ipynb          # Bowler wicket rate analysis & over-phase effects
├── RQ3_class_imbalance.ipynb          # SMOTE, class_weight, PR-AUC comparison
├── RQ4_runs_regression.ipynb          # total_runs regression: Linear, Ridge, RF, XGBoost
├── RQ5_phase_analysis.ipynb           # Powerplay/Middle/Death phase team performance
├── RQ6_dismissal_type.ipynb           # Multi-class dismissal type classification
├── RQ7_feature_engineering.ipynb      # Raw vs context-enriched feature comparison
│
├── figures/                           # All generated figures (PDF)
├── tables/                            # All generated tables (CSV)
│
├── requirements.txt                   # Required Python libraries
└── README.md                          # This file
```

---

## Research Questions

| # | Research Question | Notebook |
|---|-------------------|----------|
| RQ1 | Can ML models accurately predict whether a wicket will fall on a given delivery? | `RQ1_wicket_prediction.ipynb` |
| RQ2 | Which bowlers have the highest wicket probability, and how do over phases affect wickets? | `RQ2_bowler_analysis.ipynb` |
| RQ3 | How does class imbalance affect performance, and which strategies mitigate it best? | `RQ3_class_imbalance.ipynb` |
| RQ4 | Can regression models predict total_runs per delivery? Which features are strongest? | `RQ4_runs_regression.ipynb` |
| RQ5 | How does team performance vary across powerplay, middle, and death overs? | `RQ5_phase_analysis.ipynb` |
| RQ6 | Can a multi-class classifier distinguish between dismissal types? | `RQ6_dismissal_type.ipynb` |
| RQ7 | Do context-enriched features improve prediction over raw delivery features? | `RQ7_feature_engineering.ipynb` |

---

## Models Used

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Support Vector Machine (SVM)
- Ridge Regression (for regression tasks)

---

## How to Run on Kaggle

1. Go to [Kaggle](https://www.kaggle.com) and log in.
2. Navigate to the dataset: [IPL Ball-by-Ball Data](https://www.kaggle.com/datasets/ramjidoolla/ipl-data-set)
3. Click **"New Notebook"** on the dataset page.
4. Upload the `.ipynb` files from this repository one at a time.
5. The dataset path will be: `/kaggle/input/ipl-data-set/deliveries.csv`
6. Run all cells in order using **Run All**.
7. Figures are saved to `figures/` and tables to `tables/` in the Kaggle working directory.

> **Note:** Each notebook is self-contained. Run them independently in any order.

---

## Generated Outputs

### Figures (saved as PDF)
| File | Description | RQ |
|------|-------------|-----|
| `Figure_1_1_class_distribution.pdf` | Class imbalance bar chart | RQ1 |
| `Figure_1_2_ROC_curves.pdf` | ROC curves for all classifiers | RQ1 |
| `Figure_2_1_top_bowlers.pdf` | Top 15 bowlers by wicket rate | RQ2 |
| `Figure_2_2_phase_wicket_rate.pdf` | Wicket rate by over phase | RQ2 |
| `Figure_3_1_imbalance_comparison.pdf` | PR curves & metric comparison | RQ3 |
| `Figure_4_1_regression_results.pdf` | Predicted vs actual total_runs | RQ4 |
| `Figure_5_1_team_phase_heatmap.pdf` | Team performance heatmap | RQ5 |
| `Figure_6_1_confusion_matrix_dismissal.pdf` | Dismissal type confusion matrix | RQ6 |
| `Figure_7_1_feature_importance_comparison.pdf` | Raw vs context feature importance | RQ7 |

### Tables (saved as CSV)
| File | Description | RQ |
|------|-------------|-----|
| `Table_1_1_model_performance.csv` | All model metrics comparison | RQ1 |
| `Table_2_1_top_bowlers_wicket_rate.csv` | Top 15 bowlers stats | RQ2 |
| `Table_3_1_imbalance_strategies.csv` | Imbalance strategy comparison | RQ3 |
| `Table_4_1_regression_performance.csv` | Regression model metrics | RQ4 |
| `Table_5_1_team_phase_stats.csv` | Team-wise phase statistics | RQ5 |
| `Table_6_1_dismissal_per_class_metrics.csv` | Per-class dismissal metrics | RQ6 |
| `Table_7_1_feature_set_comparison.csv` | Raw vs context feature results | RQ7 |

---

## Key Findings

- **XGBoost and Random Forest** achieve the highest ROC-AUC for wicket prediction.
- Class imbalance (~5.5% wicket rate) requires SMOTE or `class_weight='balanced'`; raw accuracy is misleading.
- **PR-AUC** is the most reliable metric under class imbalance.
- **Context-enriched features** (cumulative runs, wickets lost, run rate) consistently improve predictions over raw delivery features.
- Wicket rates differ significantly across over phases and bowler identities.

---

## Requirements

See `requirements.txt`. Key libraries: `scikit-learn`, `xgboost`, `imbalanced-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`.
