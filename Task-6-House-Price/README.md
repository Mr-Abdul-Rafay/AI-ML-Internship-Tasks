# Task 6 - House Price Prediction

This project builds a machine learning pipeline for predicting house prices from residential property data. It includes data cleaning, feature engineering, model training, model comparison, evaluation plots, and saved model artifacts.

## Project Objective

The goal of this internship task is to train and evaluate regression models that can estimate house prices using property features such as bedrooms, bathrooms, square footage, location, year built, renovation status, view, and waterfront availability.

## Dataset

The project uses `data/data.csv`, which contains 4,600 house records with these main feature groups:

- Price target: `price`
- Property size: `sqft_living`, `sqft_lot`, `sqft_above`, `sqft_basement`
- Property details: `bedrooms`, `bathrooms`, `floors`, `condition`, `view`, `waterfront`
- Time features: `date`, `yr_built`, `yr_renovated`
- Location features: `street`, `city`, `statezip`, `country`

Invalid prices are removed before training.

## Pipeline Summary

The main script is:

```bash
python src/house_price_clean.py
```

The pipeline performs these steps:

1. Loads the house price dataset.
2. Removes invalid prices and price outliers using the IQR method.
3. Creates engineered features:
   - `house_age`
   - `renovation_age`
   - `total_rooms`
   - `total_sqft`
   - `sqft_per_room`
   - `is_renovated`
   - `has_waterfront`
   - `good_view`
4. Applies log transformation to the target variable.
5. Generates EDA visualizations.
6. Builds a preprocessing pipeline with:
   - Median imputation and scaling for numeric columns
   - Most-frequent imputation and one-hot encoding for categorical columns
7. Trains and compares multiple regression models.
8. Saves evaluation results, plots, and the best model.

## Models Used

- Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- Gradient Boosting Regressor

## Current Results

Based on the generated `outputs/model_results.csv`, the best model is Gradient Boosting.

| Model | MAE | RMSE | R2 | CV Mean |
| --- | ---: | ---: | ---: | ---: |
| Gradient Boosting | 71,733.45 | 103,621.02 | 0.7710 | 0.7431 |
| Random Forest | 80,575.18 | 117,130.39 | 0.7073 | 0.7275 |
| Lasso | 283,942.11 | 6,341,816.29 | -856.9457 | 0.5428 |
| Ridge | 649,619.86 | 17,042,565.18 | -6194.8715 | 0.5347 |
| Linear Regression | 1,521,120.07 | 42,173,967.77 | -37941.1031 | 0.2667 |

## Generated Files

After running the script, these artifacts are created:

```text
outputs/
  1_price_distribution.png
  2_correlation_heatmap.png
  3_actual_vs_predicted.png
  4_residual_analysis.png
  model_results.csv

models/
  best_house_price_model.pkl
  numerical_features.pkl
  categorical_features.pkl
```

## Installation

Create and activate a virtual environment, then install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

## How to Run

From the project root:

```bash
python src/house_price_clean.py
```

The script expects the dataset at:

```text
data/data.csv
```

## Project Structure

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
    model_results.csv
    *.png
  src/
    house_price_clean.py
  README.md
```

## Evaluation Notes

The project is well structured for an internship-level ML task because it uses a full preprocessing pipeline, compares multiple models, evaluates results with regression metrics, and saves reusable artifacts.

Important improvement areas:

- The `street` feature has very high cardinality and creates thousands of encoded columns, which can hurt efficiency and generalization.
- The script imports `GridSearchCV`, but no real hyperparameter tuning is currently performed.
- The linear models perform poorly on the final price scale, so their results should be discussed as a limitation.
- `house_age` uses a fixed year value. A better approach is to calculate age from the sale year in the `date` column.
- Cross-validation R2 is calculated on the log target, while final test R2 is calculated on the original price scale, so the two values are not directly comparable.

## Recommended Next Improvements

- Drop or simplify `street`, for example by using only `city` and `statezip`.
- Add real hyperparameter tuning for Random Forest and Gradient Boosting.
- Use sale year from `date` to calculate `house_age`.
- Save a separate cleaned dataset for reproducibility.
- Add a small prediction script that loads `models/best_house_price_model.pkl` and predicts prices for new input data.
