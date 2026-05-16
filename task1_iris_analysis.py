# =========================================================
# IRIS DATASET EXPLORATORY DATA ANALYSIS (EDA)
# AI/ML Engineering Internship - Task 1
# =========================================================
#
# Objective:
# Explore, analyze, and visualize the Iris dataset
# using professional EDA techniques.
#
# Technologies Used:
# - Python
# - Pandas
# - Matplotlib
# - Seaborn
#
# Author: Abdul Rafay
# =========================================================

# =========================
# IMPORT LIBRARIES
# =========================
import os
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ignore unnecessary warnings
warnings.filterwarnings("ignore")

# =========================
# GLOBAL SETTINGS
# =========================
sns.set_theme(style="whitegrid", palette="deep")

# Create folder for plots
OUTPUT_DIR = "eda_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# FUNCTION: LOAD DATASET
# =========================================================
def load_dataset():
    """
    Load Iris dataset using seaborn.

    Returns:
        pandas.DataFrame: Loaded iris dataset
    """
    try:
        df = sns.load_dataset("iris")

        print("=" * 60)
        print("DATASET LOADED SUCCESSFULLY")
        print("=" * 60)

        return df

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


# =========================================================
# FUNCTION: BASIC DATA INSPECTION
# =========================================================
def inspect_data(df):
    """
    Perform basic dataset inspection.
    """

    print("\n" + "=" * 60)
    print("STEP 1: BASIC DATA INSPECTION")
    print("=" * 60)

    # Dataset shape
    print(f"\nDataset Shape: {df.shape}")

    # Column names
    print("\nColumn Names:")
    print(df.columns.tolist())

    # First 5 rows
    print("\nFirst 5 Rows:")
    print(df.head())

    # Data information
    print("\nDataset Information:")
    df.info()

    # Missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Duplicate values
    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    # Unique species
    print("\nSpecies Distribution:")
    print(df["species"].value_counts())


# =========================================================
# FUNCTION: STATISTICAL SUMMARY
# =========================================================
def statistical_summary(df):
    """
    Generate statistical summary.
    """

    print("\n" + "=" * 60)
    print("STEP 2: STATISTICAL SUMMARY")
    print("=" * 60)

    print("\nDescriptive Statistics:")
    print(df.describe())

    print("\nMedian Values:")
    print(df.median(numeric_only=True))

    print("\nVariance:")
    print(df.var(numeric_only=True))

    print("\nCorrelation Matrix:")
    correlation = df.drop("species", axis=1).corr()
    print(correlation)


# =========================================================
# FUNCTION: SCATTER PLOT
# =========================================================
def scatter_plot(df):
    """
    Create scatter plot showing relationship
    between sepal length and sepal width.
    """

    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=df,
        x="sepal_length",
        y="sepal_width",
        hue="species",
        size="petal_length",
        sizes=(40, 250),
        alpha=0.8
    )

    plt.title(
        "Sepal Length vs Sepal Width",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Sepal Width (cm)")

    plt.savefig(
        f"{OUTPUT_DIR}/1_scatter_plot.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("✅ Saved: 1_scatter_plot.png")


# =========================================================
# FUNCTION: PAIRPLOT
# =========================================================
def pair_plot(df):
    """
    Create pairplot for all numerical features.
    """

    pairplot = sns.pairplot(
        df,
        hue="species",
        diag_kind="hist",
        corner=False
    )

    pairplot.fig.suptitle(
        "Pairwise Relationships Between Features",
        fontsize=18,
        fontweight="bold",
        y=1.02
    )

    pairplot.savefig(
        f"{OUTPUT_DIR}/2_pairplot.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("✅ Saved: 2_pairplot.png")


# =========================================================
# FUNCTION: HISTOGRAMS
# =========================================================
def histogram_plots(df):
    """
    Create histograms for all numerical features.
    """

    numerical_columns = df.select_dtypes(include=["float64"]).columns

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes = axes.flatten()

    for i, column in enumerate(numerical_columns):

        sns.histplot(
            data=df,
            x=column,
            kde=True,
            bins=20,
            ax=axes[i]
        )

        axes[i].set_title(
            f"Distribution of {column}",
            fontsize=13,
            fontweight="bold"
        )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/3_histograms.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("✅ Saved: 3_histograms.png")


# =========================================================
# FUNCTION: BOXPLOTS
# =========================================================
def box_plots(df):
    """
    Create boxplots to identify outliers.
    """

    features = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes = axes.flatten()

    for i, feature in enumerate(features):

        sns.boxplot(
            data=df,
            x="species",
            y=feature,
            ax=axes[i]
        )

        axes[i].set_title(
            f"{feature} by Species",
            fontsize=13,
            fontweight="bold"
        )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/4_boxplots.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("✅ Saved: 4_boxplots.png")


# =========================================================
# FUNCTION: CORRELATION HEATMAP
# =========================================================
def correlation_heatmap(df):
    """
    Create heatmap showing feature correlations.
    """

    correlation = df.drop("species", axis=1).corr()

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        linewidths=1,
        fmt=".2f",
        square=True
    )

    plt.title(
        "Feature Correlation Heatmap",
        fontsize=16,
        fontweight="bold"
    )

    plt.savefig(
        f"{OUTPUT_DIR}/5_heatmap.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("✅ Saved: 5_heatmap.png")


# =========================================================
# FUNCTION: GENERATE INSIGHTS
# =========================================================
def generate_insights(df):
    """
    Print analytical insights from the dataset.
    """

    print("\n" + "=" * 60)
    print("STEP 4: KEY ANALYTICAL INSIGHTS")
    print("=" * 60)

    correlation = df.drop("species", axis=1).corr()

    petal_corr = correlation.loc["petal_length", "petal_width"]

    insights = [
        "1. The Iris dataset contains 150 observations and 5 columns.",
        
        "2. The dataset contains no missing values, making it clean and ready for machine learning tasks.",

        "3. Setosa species is clearly separable from Versicolor and Virginica based on petal dimensions.",

        f"4. Petal length and petal width show a very strong positive correlation ({petal_corr:.2f}), indicating possible multicollinearity.",

        "5. Sepal width contains the highest number of outliers among all numerical features.",

        "6. Petal-based features are more useful for species classification compared to sepal-based features.",

        "7. The dataset is balanced because each species contains an equal number of samples.",

        "8. Feature distributions are approximately normal with slight variations across species."
    ]

    for insight in insights:
        print(insight)


# =========================================================
# FUNCTION: MAIN EXECUTION
# =========================================================
def main():

    # Load dataset
    iris = load_dataset()

    if iris is None:
        return

    # Data inspection
    inspect_data(iris)

    # Statistical summary
    statistical_summary(iris)

    print("\n" + "=" * 60)
    print("STEP 3: DATA VISUALIZATION")
    print("=" * 60)

    # Generate plots
    scatter_plot(iris)
    pair_plot(iris)
    histogram_plots(iris)
    box_plots(iris)
    correlation_heatmap(iris)

    # Generate insights
    generate_insights(iris)

    print("\n" + "=" * 60)
    print("EDA TASK COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"\nAll visualizations saved inside '{OUTPUT_DIR}' folder.")


# =========================================================
# PROGRAM ENTRY POINT
# =========================================================
if __name__ == "__main__":
    main()