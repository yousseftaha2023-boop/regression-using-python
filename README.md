
# Medical Insurance Charges — Regression Analysis

Predicts annual medical insurance charges from patient demographics (age, BMI, children, smoking status, region) using linear regression, with statistical feature selection and a KNN benchmark comparison.

**Dataset:** 1,338 records · 7 features (medical-charges dataset)

## Results

| Model | Test R² |
|---|---|
| **Linear Regression** | **0.781** |
| KNN Regressor | 0.189 |

![Predicted vs Actual](pred_vs_actual.png)

![Model Comparison](r2_comparison.png)

**Read on results:** Linear Regression explains ~78% of the variance in medical charges using just 4 features (age, BMI, children, smoker status) — smoker status alone is the dominant driver, visible in the two distinct diagonal bands in the predicted-vs-actual plot. KNN underperforms badly here (R² = 0.19) because its inputs weren't feature-scaled before fitting — distance-based models like KNN need normalized features to work properly, and this is a known limitation of the current version worth fixing.

## Pipeline

- **EDA:** distribution analysis, skewness checks, and category proportions across age, BMI, children, smoker, and region
- **Statistical testing:** one-way ANOVA on categorical predictors (smoker, region, sex) against charges
- **Feature selection:** compared RFECV, forward selection, and backward elimination (all converged on `age`, `bmi`, `children`, `smoker`)
- **Modeling:** OLS regression (statsmodels) for interpretability + scikit-learn LinearRegression for train/test evaluation
- **Benchmark comparison:** KNN Regressor with 5-fold cross-validation

## Known limitation

KNN was run without feature scaling (StandardScaler was imported but not applied before fitting) — this understates KNN's real performance and is a planned fix.

## Tools used

Python · Pandas · scikit-learn · statsmodels · SciPy · Plotly · Matplotlib · Seaborn

## Project structure

```
insaurance (regression).py     # Full pipeline: EDA, ANOVA, feature selection, modeling
pred_vs_actual.png             # Predicted vs actual charges (Linear Regression)
r2_comparison.png              # R² comparison across models
```

## How to run

```bash
pip install pandas scikit-learn statsmodels scipy plotly matplotlib seaborn
python "insaurance (regression).py"
```

## Author

Youssef Taha — Statistics undergraduate, Cairo University
