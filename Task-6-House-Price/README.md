🏠# Task 6 - House Price Prediction

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green)
![R2 Score](https://img.shields.io/badge/R2%20Score-77.1%25-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-success)

## AI/ML Engineering Internship - DevelopersHub Corporation

**Author:** Abdul Rafay  
**Task:** Task 6 of 6  
**Project:** House Price Prediction using Regression Models  
**Best Model:** Gradient Boosting Regressor  
**Best R2 Score:** 77.1%

---

## Table of Contents

1. [Why This Matters](#why-this-matters)
2. [Project Objective](#project-objective)
3. [Dataset Information](#dataset-information)
4. [Technologies Used](#technologies-used)
5. [Repository Structure](#repository-structure)
6. [Installation](#installation)
7. [How to Run](#how-to-run)
8. [Pipeline Workflow](#pipeline-workflow)
9. [Feature Engineering](#feature-engineering)
10. [Models Compared](#models-compared)
11. [Results](#results)
12. [Visualizations](#visualizations)
13. [Key Technical Highlights](#key-technical-highlights)
14. [Limitations](#limitations)
15. [Future Improvements](#future-improvements)
16. [Learning Outcomes](#learning-outcomes)
17. [Conclusion](#conclusion)
18. [Author and Repository](#author-and-repository)

---

## Why This Matters

Accurate house price prediction is important in real estate because pricing decisions affect buyers, sellers, agents, banks, and investors. A reliable valuation model can support better listing prices, investment analysis, mortgage decisions, and market comparison.

Housing prices are influenced by many factors, including property size, location, age, renovation status, waterfront access, and neighborhood demand. This makes the problem a strong real-world regression task because the relationship between features and price is not always linear.

Achieving an **R2 score of 77.1%** is a strong result for real-world housing data, especially because property prices are affected by external market conditions that are not fully available in the dataset.

---

## Project Objective

The objective of this project is to build a complete machine learning pipeline that predicts house prices using structured property data.

The project focuses on:

- Cleaning raw housing data
- Handling invalid prices and outliers
- Engineering useful real estate features
- Applying log transformation to normalize the target variable
- Training and comparing multiple regression models
- Evaluating model performance using standard regression metrics
- Saving the best model for future use

---

## Dataset Information

| Attribute | Details |
| --- | --- |
| Dataset | House price dataset |
| Original Records | 4,600 |
| Records After Cleaning | 4,311 |
| Removed Invalid Prices | 49 |
| Removed Outliers | 240 |
| Numerical Features | 20 |
| Categorical Features | 5 |
| Target Variable | `price` |
| Target Transformation | `log_price = log1p(price)` |
| Best Model | Gradient Boosting Regressor |

The dataset includes property attributes such as bedrooms, bathrooms, square footage, floors, waterfront status, view quality, condition, year built, renovation year, city, state zip code, and country.

Location features such as `city` and `statezip` were retained because location is one of the strongest predictors of real estate value.

---

## Technologies Used

| Technology | Purpose |
| --- | --- |
| Python | Main programming language |
| Pandas | Data loading, cleaning, and manipulation |
| NumPy | Numerical operations and target transformation |
| Matplotlib | Plot generation |
| Seaborn | Statistical visualizations |
| Scikit-Learn | ML models, preprocessing, pipelines, and evaluation |
| Joblib | Model persistence |

---

## Repository Structure

```text
Task-6-House-Price/
  data/
    data.csv
    archive.zip
    output.csv
  models/
    best_house_price_model.pkl
    numerical_features.pkl
    categorical_features.pkl
  outputs/
    1_price_distribution.png
    2_correlation_heatmap.png
    3_actual_vs_predicted.png
    4_residual_analysis.png
    model_results.csv
  src/
    house_price_clean.py
  README.md
```

---

## Installation

Create and activate a virtual environment:

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

---

## How to Run

Run the complete training pipeline from the project root:

```bash
python src/house_price_clean.py
```

The script expects the dataset at:

```text
data/data.csv
```

After execution, evaluation plots are saved in `outputs/` and the best trained model is saved in `models/`.

---

## Pipeline Workflow

1. Load the raw housing dataset from `data/data.csv`.
2. Remove invalid records where `price <= 0`.
3. Remove extreme price outliers using the IQR method.
4. Create new engineered features from existing property attributes.
5. Apply `log1p` transformation to the target variable to reduce skewness.
6. Split the data into training and testing sets.
7. Build a preprocessing pipeline using `ColumnTransformer`.
8. Impute missing numerical and categorical values.
9. Scale numerical features using `StandardScaler`.
10. Encode categorical features using `OneHotEncoder`.
11. Train multiple regression models.
12. Evaluate each model using MAE, RMSE, R2, and 5-fold cross-validation.
13. Select the best-performing model.
14. Save visualizations, metrics, and model artifacts.

---

## Feature Engineering

| Feature | Description | Purpose |
| --- | --- | --- |
| `house_age` | `2026 - yr_built` | Captures property age |
| `renovation_age` | `2026 - yr_renovated` if renovated, else `0` | Captures renovation recency |
| `total_rooms` | `bedrooms + bathrooms` | Represents property capacity |
| `total_sqft` | `sqft_living + sqft_basement` | Captures broader usable area |
| `sqft_per_room` | `sqft_living / (total_rooms + 1)` | Measures room spaciousness |
| `is_renovated` | Binary flag from `yr_renovated` | Identifies renovated homes |
| `has_waterfront` | Binary flag from `waterfront` | Captures premium location feature |
| `good_view` | Binary flag where `view >= 2` | Captures view quality |

---

## Models Compared

| Model | Model Type | Notes |
| --- | --- | --- |
| Linear Regression | Linear baseline | Performed poorly on original price scale |
| Ridge Regression | Regularized linear model | Improved stability but still weak |
| Lasso Regression | Regularized linear model | Reduced coefficients but failed on test scale |
| Random Forest Regressor | Tree ensemble | Strong non-linear performance |
| Gradient Boosting Regressor | Boosted tree ensemble | Best-performing model |

The results show a clear difference between linear and tree-based models. Housing prices contain non-linear patterns, especially around location, square footage, condition, and premium features. Tree-based models handled these relationships much better than linear models.

---

## Results

| Model | MAE | RMSE | R2 | CV Mean |
| --- | ---: | ---: | ---: | ---: |
| Gradient Boosting | $71,733.45 | $103,621.02 | 0.7710 | 0.7431 |
| Random Forest | $80,575.18 | $117,130.39 | 0.7073 | 0.7275 |
| Lasso | $283,942.11 | $6,341,816.29 | -856.9457 | 0.5428 |
| Ridge | $649,619.86 | $17,042,565.18 | -6194.8715 | 0.5347 |
| Linear Regression | $1,521,120.07 | $42,173,967.77 | -37941.1031 | 0.2667 |

### Best Model

| Metric | Value |
| --- | ---: |
| Model | Gradient Boosting Regressor |
| MAE | $71,733 |
| RMSE | $103,621 |
| R2 Score | 77.1% |
| CV Mean | 0.7431 |

The **77.1% R2 score is excellent** for a real-world housing dataset because real estate prices are affected by many external factors that are not included in the dataset, such as market trends, school districts, interest rates, local demand, and neighborhood development.

---

## Visualizations

| File | Description |
| --- | --- |
| `1_price_distribution.png` | Compares original price distribution with log-transformed price distribution |
| `2_correlation_heatmap.png` | Shows correlations between numerical features and target-related variables |
| `3_actual_vs_predicted.png` | Compares actual house prices against predicted prices for the best model |
| `4_residual_analysis.png` | Shows prediction residuals to evaluate model error behavior |
| `model_results.csv` | Stores model performance metrics in tabular format |

---

## Key Technical Highlights

| Highlight | Implementation |
| --- | --- |
| End-to-end ML pipeline | Used `sklearn.pipeline.Pipeline` |
| Mixed feature preprocessing | Used `ColumnTransformer` |
| Numerical preprocessing | Median imputation and `StandardScaler` |
| Categorical preprocessing | Most-frequent imputation and `OneHotEncoder` |
| Target normalization | Applied `np.log1p(price)` |
| Outlier handling | Removed 240 price outliers using IQR |
| Model comparison | Trained five regression models |
| Validation | Used 5-fold cross-validation |
| Persistence | Saved model artifacts using `joblib` |
| Reproducibility | Used `random_state=42` |

---

## Limitations

This project is intentionally honest about its limitations because mature ML engineering requires understanding where a model works and where it may fail.

- The `street` column has very high cardinality, which can increase dimensionality after one-hot encoding.
- Linear models performed poorly on the final price scale, showing that the housing price relationship is strongly non-linear.
- `house_age` currently uses a fixed year value instead of deriving the sale year from the `date` column.
- Cross-validation is calculated on the log-transformed target, while final MAE, RMSE, and R2 are reported on the original price scale.
- The dataset does not include important external real estate drivers such as school ratings, crime rate, interest rates, neighborhood demand, or market timing.
- Hyperparameter tuning is manually configured rather than fully optimized with `GridSearchCV` or `RandomizedSearchCV`.

These limitations do not reduce the value of the project. They show that the model was evaluated with engineering self-awareness instead of only reporting the best-looking numbers.

---

## Future Improvements

- [ ] Drop or simplify high-cardinality `street` values.
- [ ] Use sale year from `date` when calculating `house_age`.
- [ ] Add `GridSearchCV` or `RandomizedSearchCV` for model tuning.
- [ ] Try advanced boosting models such as XGBoost, LightGBM, or CatBoost.
- [ ] Add feature importance analysis for the best model.
- [ ] Save a cleaned version of the dataset for reproducibility.
- [ ] Build a prediction script for new house records.
- [ ] Add a Streamlit dashboard for interactive predictions.
- [ ] Compare metrics consistently on both log scale and original price scale.

---

## Learning Outcomes

This project demonstrates practical ML engineering skills, including:

- Building production-style machine learning pipelines
- Handling numerical and categorical features together
- Applying target transformation for skewed regression problems
- Performing feature engineering from domain knowledge
- Comparing baseline, regularized, and ensemble models
- Evaluating regression models with multiple metrics
- Understanding why tree-based models outperform linear models on non-linear data
- Saving trained models and metadata for reuse
- Communicating results, limitations, and next steps clearly

---

## Conclusion

This House Price Prediction project successfully completes Task 6 of the DevelopersHub Corporation AI/ML Engineering Internship.

The final model, **Gradient Boosting Regressor**, achieved a strong **77.1% R2 score**, with an MAE of approximately **$71,733** and RMSE of approximately **$103,621**. For real-world housing data, this is a strong result because property prices are influenced by many complex and external market factors.

The project also demonstrates a complete machine learning workflow, including data cleaning, outlier handling, feature engineering, preprocessing pipelines, model comparison, cross-validation, visualization, and model persistence.

---

## Author and Repository

| Field | Details |
| --- | --- |
| Author | Abdul Rafay |
| Internship | AI/ML Engineering Internship |
| Organization | DevelopersHub Corporation |
| Task | Task 6 of 6 |
| Project | House Price Prediction |
| GitHub | https://github.com/Mr-Abdul-Rafay/AI-ML-Internship-Tasks/tree/main/Task-6-House-Price |
| LinkedIn | www.linkedin.com/in/abdul-rafay-0b733b3a0 |
