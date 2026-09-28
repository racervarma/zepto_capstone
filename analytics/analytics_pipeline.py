"""
Zepto Capstone - Analytics Pipeline

This script:
1. Loads the Titanic dataset exactly once using seaborn.
2. Saves the raw dataset immediately as analytics/titanic.csv.
3. Profiles and cleans the dataset.
4. Performs univariate, bivariate, correlation and data-story analysis.
5. Performs a standardization sanity check.
6. Builds classification pipelines for Logistic Regression,
   Decision Tree and Random Forest.
7. Compares imbalance strategies.
8. Tunes a Random Forest using GridSearchCV.
9. Builds a multivariable regression model for fare.
10. Saves the best complete classification pipeline with joblib.
11. Reloads the saved pipeline and predicts on raw data.

All generated artifacts are written inside the analytics/ folder.
"""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.base import clone

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

ANALYTICS_DIR = Path(__file__).resolve().parent
CSV_PATH = ANALYTICS_DIR / "titanic.csv"

UNIVARIATE_PATH = ANALYTICS_DIR / "univariate.png"
CORR_PATH = ANALYTICS_DIR / "corr_heatmap.png"

STORY_PATHS = [
    ANALYTICS_DIR / "story_1.png",
    ANALYTICS_DIR / "story_2.png",
    ANALYTICS_DIR / "story_3.png",
    ANALYTICS_DIR / "story_4.png",
]

TREE_PATH = ANALYTICS_DIR / "tree.png"
RESIDUAL_PATH = ANALYTICS_DIR / "residuals.png"
MODEL_PATH = ANALYTICS_DIR / "model_pipeline.pkl"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def calculate_iqr_outliers(series):
    """Return IQR boundaries and number of outliers."""
    clean = series.dropna()

    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = clean[(clean < lower) | (clean > upper)]

    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower": lower,
        "upper": upper,
        "count": len(outliers),
    }


def survival_rate_table(df, columns):
    """
    Calculate survival rate using boolean masking.

    This intentionally uses boolean filtering rather than
    relying only on groupby.
    """
    rows = []

    if columns == ["sex"]:
        for value in sorted(df["sex"].dropna().unique()):
            mask = df["sex"] == value
            rate = df.loc[mask, "survived"].mean()
            rows.append({"sex": value, "survival_rate": rate})

    elif columns == ["pclass"]:
        for value in sorted(df["pclass"].dropna().unique()):
            mask = df["pclass"] == value
            rate = df.loc[mask, "survived"].mean()
            rows.append({"pclass": value, "survival_rate": rate})

    elif columns == ["sex", "pclass"]:
        sexes = sorted(df["sex"].dropna().unique())
        classes = sorted(df["pclass"].dropna().unique())

        for sex in sexes:
            for pclass in classes:
                mask = (df["sex"] == sex) & (df["pclass"] == pclass)
                subset = df.loc[mask]

                if len(subset) > 0:
                    rate = subset["survived"].mean()
                    rows.append(
                        {
                            "sex": sex,
                            "pclass": pclass,
                            "survival_rate": rate,
                        }
                    )

    return pd.DataFrame(rows)


def evaluate_classifier(name, pipeline, X_test, y_test):
    """Evaluate a fitted classification pipeline."""
    predictions = pipeline.predict(X_test)

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X_test)[:, 1]
    else:
        probabilities = pipeline.decision_function(X_test)

    cm = confusion_matrix(y_test, predictions)

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(y_test, probabilities),
        "Confusion_Matrix": cm,
    }

    return metrics


def adjusted_r2(r2, n, p):
    """Calculate adjusted R-squared."""
    if n <= p + 1:
        return np.nan

    return 1 - ((1 - r2) * (n - 1) / (n - p - 1))


# ============================================================
# MAIN
# ============================================================

def main():

    print_section("PART A - DATA LOADING")

    # --------------------------------------------------------
    # IMPORTANT:
    # Load Titanic exactly ONCE.
    # --------------------------------------------------------

    df = sns.load_dataset("titanic")

    # Immediately save raw loaded dataset.
    df.to_csv(CSV_PATH, index=False)

    print(f"Raw dataset saved to: {CSV_PATH}")
    print(f"Shape immediately after loading: {df.shape}")

    # --------------------------------------------------------
    # Profiling
    # --------------------------------------------------------

    print_section("DATA INFO")

    df.info()

    print_section("DESCRIPTIVE STATISTICS")

    print(df.describe(include="all"))

    print_section("DATA SHAPE")

    print(df.shape)

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print_section("MISSING VALUE ANALYSIS")

    missing_percentages = (
        df.isna().mean().mul(100).sort_values(ascending=False)
    )

    missing_table = missing_percentages[
        missing_percentages > 0
    ].to_frame("missing_percentage")

    print(missing_table)

    # --------------------------------------------------------
    # Missing-value cleaning
    #
    # Rule:
    # < 5%       -> drop rows
    # 5%-30%     -> impute
    # > 30%      -> drop column OR encode missing
    #
    # For this dataset:
    # deck is >30%, so we encode missing as "missing".
    # This preserves the variable for analysis without
    # unreliable numeric imputation.
    # --------------------------------------------------------

    print_section("CLEANING")

    cleaned = df.copy()

    for column in df.columns:

        missing_rate = df[column].isna().mean() * 100

        if missing_rate == 0:
            continue

        if missing_rate < 5:

            print(
                f"{column}: {missing_rate:.2f}% missing -> "
                "drop affected rows"
            )

            cleaned = cleaned.dropna(subset=[column])

        elif missing_rate <= 30:

            print(
                f"{column}: {missing_rate:.2f}% missing -> "
                "impute"
            )

            if pd.api.types.is_numeric_dtype(cleaned[column]):
                cleaned[column] = cleaned[column].fillna(
                    cleaned[column].median()
                )
            else:
                mode = cleaned[column].mode(dropna=True)

                if len(mode) > 0:

                    # Categorical columns need the mode to
                    # already exist as a category, which it does.
                    cleaned[column] = cleaned[column].fillna(
                        mode.iloc[0]
                    )

                else:
                    if isinstance(
                        cleaned[column].dtype,
                        pd.CategoricalDtype
                    ):
                        if "missing" not in cleaned[column].cat.categories:
                            cleaned[column] = (
                                cleaned[column]
                                .cat.add_categories(["missing"])
                            )

                    cleaned[column] = cleaned[column].fillna("missing")

        else:

            print(
                f"{column}: {missing_rate:.2f}% missing -> "
                "encode missing as 'missing'"
            )

            if pd.api.types.is_numeric_dtype(cleaned[column]):

                cleaned[column] = cleaned[column].fillna(
                    cleaned[column].median()
                )

            else:

                # Important for categorical columns such as `deck`.
                if isinstance(
                    cleaned[column].dtype,
                    pd.CategoricalDtype
                ):
                    if "missing" not in cleaned[column].cat.categories:
                        cleaned[column] = (
                            cleaned[column]
                            .cat.add_categories(["missing"])
                        )

                cleaned[column] = cleaned[column].fillna("missing")

    print("\nCleaned shape:", cleaned.shape)

    # ========================================================
    # UNIVARIATE ANALYSIS
    # ========================================================

    print_section("UNIVARIATE ANALYSIS")

    age_outliers = calculate_iqr_outliers(cleaned["age"])
    fare_outliers = calculate_iqr_outliers(cleaned["fare"])

    print("Age IQR results:")
    print(age_outliers)

    print("\nFare IQR results:")
    print(fare_outliers)

    fare_mean = cleaned["fare"].mean()
    fare_median = cleaned["fare"].median()

    fare_modes = cleaned["fare"].mode()

    print(f"\nFare mean: {fare_mean:.4f}")
    print(f"Fare median: {fare_median:.4f}")
    print(f"Fare mode: {fare_modes.tolist()}")

    # Histogram + boxplot for age and fare.
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.histplot(
        cleaned["age"],
        kde=True,
        ax=axes[0, 0],
        color="steelblue",
    )
    axes[0, 0].set_title("Age Distribution")

    sns.boxplot(
        x=cleaned["age"],
        ax=axes[0, 1],
        color="skyblue",
    )
    axes[0, 1].set_title("Age Boxplot")

    sns.histplot(
        cleaned["fare"],
        kde=True,
        ax=axes[1, 0],
        color="darkorange",
    )
    axes[1, 0].set_title("Fare Distribution")

    sns.boxplot(
        x=cleaned["fare"],
        ax=axes[1, 1],
        color="orange",
    )
    axes[1, 1].set_title("Fare Boxplot")

    plt.tight_layout()
    plt.savefig(UNIVARIATE_PATH, dpi=150)
    plt.close()

    print(f"Saved: {UNIVARIATE_PATH}")

    # ========================================================
    # BIVARIATE ANALYSIS
    # ========================================================

    print_section("BIVARIATE SURVIVAL ANALYSIS")

    sex_rates = survival_rate_table(cleaned, ["sex"])
    pclass_rates = survival_rate_table(cleaned, ["pclass"])
    sex_pclass_rates = survival_rate_table(
        cleaned,
        ["sex", "pclass"],
    )

    print("\nSurvival rate by sex:")
    print(sex_rates.to_string(index=False))

    print("\nSurvival rate by pclass:")
    print(pclass_rates.to_string(index=False))

    print("\nSurvival rate by sex + pclass:")
    print(sex_pclass_rates.to_string(index=False))

    # ========================================================
    # CORRELATION
    # ========================================================

    print_section("CORRELATION ANALYSIS")

    correlation_columns = [
        "survived",
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare",
    ]

    corr = cleaned[correlation_columns].corr()

    print(corr)

    plt.figure(figsize=(9, 7))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
    )

    plt.title("Titanic Numeric Correlation Matrix")
    plt.tight_layout()
    plt.savefig(CORR_PATH, dpi=150)
    plt.close()

    print(f"Saved: {CORR_PATH}")

    # Find strongest two absolute off-diagonal correlations.
    pairs = []

    for i in range(len(correlation_columns)):
        for j in range(i + 1, len(correlation_columns)):

            col1 = correlation_columns[i]
            col2 = correlation_columns[j]

            value = corr.loc[col1, col2]

            pairs.append(
                {
                    "feature_1": col1,
                    "feature_2": col2,
                    "correlation": value,
                    "absolute_correlation": abs(value),
                }
            )

    pairs_df = pd.DataFrame(pairs)

    top_two = pairs_df.sort_values(
        "absolute_correlation",
        ascending=False,
    ).head(2)

    print("\nTop 2 strongest correlation pairs:")
    print(top_two.to_string(index=False))

    # ========================================================
    # DATA STORY
    # ========================================================

    print_section("DATA STORY")

    # Story 1 - survival by sex
    plt.figure(figsize=(8, 6))

    story1 = (
        cleaned.groupby("sex")["survived"]
        .mean()
        .reset_index()
    )

    sns.barplot(
        data=story1,
        x="sex",
        y="survived",
        hue="sex",
        palette="Set2",
        legend=False,
    )

    plt.ylabel("Survival Rate")
    plt.title("Story 1: Survival Rate by Sex")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(STORY_PATHS[0], dpi=150)
    plt.close()

    # Story 2 - survival by class
    plt.figure(figsize=(8, 6))

    story2 = (
        cleaned.groupby("pclass")["survived"]
        .mean()
        .reset_index()
    )

    sns.barplot(
        data=story2,
        x="pclass",
        y="survived",
        hue="pclass",
        palette="viridis",
        legend=False,
    )

    plt.ylabel("Survival Rate")
    plt.xlabel("Passenger Class")
    plt.title("Story 2: Survival Rate by Passenger Class")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(STORY_PATHS[1], dpi=150)
    plt.close()

    # Story 3 - sex and class
    plt.figure(figsize=(9, 6))

    story3 = (
        cleaned.groupby(
            ["sex", "pclass"],
            as_index=False,
        )["survived"]
        .mean()
    )

    sns.barplot(
        data=story3,
        x="pclass",
        y="survived",
        hue="sex",
        palette="Set1",
    )

    plt.ylabel("Survival Rate")
    plt.xlabel("Passenger Class")
    plt.title("Story 3: Survival by Sex and Passenger Class")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(STORY_PATHS[2], dpi=150)
    plt.close()

    # Story 4 - age vs fare
    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=cleaned,
        x="age",
        y="fare",
        hue="survived",
        alpha=0.65,
        palette={0: "red", 1: "green"},
    )

    plt.title("Story 4: Age, Fare and Survival")
    plt.xlabel("Age")
    plt.ylabel("Fare")
    plt.tight_layout()
    plt.savefig(STORY_PATHS[3], dpi=150)
    plt.close()

    for path in STORY_PATHS:
        print(f"Saved: {path}")

    # ========================================================
    # STANDARDIZATION CHECK
    # ========================================================

    print_section("STANDARDIZATION CHECK")

    standardization_df = cleaned.copy()

    before_age_mean = standardization_df["age"].mean()
    before_age_std = standardization_df["age"].std()

    before_fare_mean = standardization_df["fare"].mean()
    before_fare_std = standardization_df["fare"].std()

    standardization_df["age_z"] = (
        standardization_df["age"] - before_age_mean
    ) / before_age_std

    standardization_df["fare_z"] = (
        standardization_df["fare"] - before_fare_mean
    ) / before_fare_std

    print("BEFORE")
    print(
        f"Age  mean={before_age_mean:.6f}, "
        f"std={before_age_std:.6f}"
    )
    print(
        f"Fare mean={before_fare_mean:.6f}, "
        f"std={before_fare_std:.6f}"
    )

    print("\nAFTER")

    print(
        f"Age z-score  mean={standardization_df['age_z'].mean():.6f}, "
        f"std={standardization_df['age_z'].std():.6f}"
    )

    print(
        f"Fare z-score mean={standardization_df['fare_z'].mean():.6f}, "
        f"std={standardization_df['fare_z'].std():.6f}"
    )

    # ========================================================
    # PART B - CLASSIFICATION
    # ========================================================

    print_section("PART B - PREDICTIVE MODELING")

    # Features selected for classification.
    #
    # survived is target.
    # We exclude alive because it directly duplicates survived.
    # We also exclude redundant boolean flags adult_male and alone.
    #
    # The model therefore uses:
    # pclass, sex, age, sibsp, parch, fare, embarked,
    # class, who, deck, embark_town.
    #
    # However, to keep the model clean and avoid redundant
    # representations, we use the main requested variables.
    # --------------------------------------------------------

    classification_features = [
        "pclass",
        "sex",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked",
    ]

    X = cleaned[classification_features].copy()
    y = cleaned["survived"].copy()

    print("\nClass distribution:")
    print(y.value_counts())
    print("\nClass proportions:")
    print(y.value_counts(normalize=True))

    # --------------------------------------------------------
    # Stratified train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print("\nTrain shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    numeric_features = [
        "pclass",
        "age",
        "sibsp",
        "parch",
        "fare",
    ]

    categorical_features = [
        "sex",
        "embarked",
    ]

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numeric_features,
            ),
            (
                "cat",
                categorical_transformer,
                categorical_features,
            ),
        ]
    )

    # ========================================================
    # THREE CLASSIFIERS
    # ========================================================

    classifiers = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
    }

    fitted_models = {}
    classification_results = []

    for name, estimator in classifiers.items():

        print_section(f"TRAINING {name}")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", estimator),
            ]
        )

        pipeline.fit(X_train, y_train)

        fitted_models[name] = pipeline

        result = evaluate_classifier(
            name,
            pipeline,
            X_test,
            y_test,
        )

        classification_results.append(result)

        print("Confusion matrix:")
        print(result["Confusion_Matrix"])

        print(f"Accuracy:  {result['Accuracy']:.4f}")
        print(f"Precision: {result['Precision']:.4f}")
        print(f"Recall:    {result['Recall']:.4f}")
        print(f"F1:        {result['F1']:.4f}")
        print(f"ROC AUC:   {result['ROC_AUC']:.4f}")

    classification_table = pd.DataFrame(
        [
            {
                "Model": r["Model"],
                "Accuracy": r["Accuracy"],
                "Precision": r["Precision"],
                "Recall": r["Recall"],
                "F1": r["F1"],
                "ROC_AUC": r["ROC_AUC"],
            }
            for r in classification_results
        ]
    )

    print_section("CLASSIFICATION COMPARISON")

    print(
        classification_table.to_string(
            index=False
        )
    )

    # ========================================================
    # DECISION TREE VISUALIZATION
    # ========================================================

    print_section("DECISION TREE VISUALIZATION")

    decision_tree_pipeline = fitted_models["Decision Tree"]

    tree_model = decision_tree_pipeline.named_steps["model"]
    fitted_preprocessor = (
        decision_tree_pipeline.named_steps["preprocessor"]
    )

    try:
        feature_names = (
            fitted_preprocessor.get_feature_names_out()
        )
    except Exception:
        feature_names = [
            f"feature_{i}"
            for i in range(tree_model.n_features_in_)
        ]

    plt.figure(figsize=(24, 14))

    plot_tree(
        tree_model,
        feature_names=feature_names,
        class_names=["Did not survive", "Survived"],
        filled=True,
        rounded=True,
        max_depth=4,
        fontsize=8,
    )

    plt.title("Decision Tree")
    plt.tight_layout()
    plt.savefig(TREE_PATH, dpi=150)
    plt.close()

    print(f"Saved: {TREE_PATH}")

    # ========================================================
    # IMBALANCE COMPARISON
    # ========================================================

    print_section("IMBALANCE STRATEGY COMPARISON")

    # Baseline
    baseline_rf = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    baseline_rf.fit(X_train, y_train)

    baseline_result = evaluate_classifier(
        "Baseline RF",
        baseline_rf,
        X_test,
        y_test,
    )

    # class_weight balanced
    balanced_rf = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    balanced_rf.fit(X_train, y_train)

    balanced_result = evaluate_classifier(
        "Balanced RF",
        balanced_rf,
        X_test,
        y_test,
    )

    # SMOTE
    smote_pipeline = ImbPipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            (
                "smote",
                SMOTE(random_state=42),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    smote_pipeline.fit(X_train, y_train)

    smote_result = evaluate_classifier(
        "SMOTE RF",
        smote_pipeline,
        X_test,
        y_test,
    )

    imbalance_results = pd.DataFrame(
        [
            {
                "Strategy": baseline_result["Model"],
                "Precision": baseline_result["Precision"],
                "Recall": baseline_result["Recall"],
                "F1": baseline_result["F1"],
            },
            {
                "Strategy": balanced_result["Model"],
                "Precision": balanced_result["Precision"],
                "Recall": balanced_result["Recall"],
                "F1": balanced_result["F1"],
            },
            {
                "Strategy": smote_result["Model"],
                "Precision": smote_result["Precision"],
                "Recall": smote_result["Recall"],
                "F1": smote_result["F1"],
            },
        ]
    )

    print(imbalance_results.to_string(index=False))

    # ========================================================
    # GRID SEARCH RANDOM FOREST
    # ========================================================

    print_section("RANDOM FOREST GRID SEARCH")

    grid_pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            (
                "model",
                RandomForestClassifier(
                    oob_score=True,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 5, 10],
        "model__max_features": ["sqrt", "log2"],
    }

    grid_search = GridSearchCV(
        estimator=grid_pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        refit=True,
    )

    grid_search.fit(X_train, y_train)

    print("Best parameters:")
    print(grid_search.best_params_)

    best_rf_pipeline = grid_search.best_estimator_

    best_rf_model = best_rf_pipeline.named_steps["model"]

    print(
        f"OOB score: {best_rf_model.oob_score_:.6f}"
    )

    tuned_result = evaluate_classifier(
        "Tuned Random Forest",
        best_rf_pipeline,
        X_test,
        y_test,
    )

    print("\nTuned Random Forest test metrics:")

    for metric in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC_AUC",
    ]:
        print(
            f"{metric}: "
            f"{tuned_result[metric]:.4f}"
        )

    # ========================================================
    # REGRESSION
    # Predict FARE from other available features.
    # ========================================================

    print_section("FARE REGRESSION")

    regression_features = [
        "pclass",
        "sex",
        "age",
        "sibsp",
        "parch",
        "embarked",
    ]

    X_reg = cleaned[regression_features].copy()
    y_reg = cleaned["fare"].copy()

    (
        X_reg_train,
        X_reg_test,
        y_reg_train,
        y_reg_test,
    ) = train_test_split(
        X_reg,
        y_reg,
        test_size=0.20,
        random_state=42,
    )

    regression_numeric = [
        "pclass",
        "age",
        "sibsp",
        "parch",
    ]

    regression_categorical = [
        "sex",
        "embarked",
    ]

    regression_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        ),
                    ]
                ),
                regression_numeric,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                regression_categorical,
            ),
        ]
    )

    regression_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                regression_preprocessor,
            ),
            (
                "model",
                LinearRegression(),
            ),
        ]
    )

    regression_pipeline.fit(
        X_reg_train,
        y_reg_train,
    )

    regression_predictions = regression_pipeline.predict(
        X_reg_test
    )

    mae = mean_absolute_error(
        y_reg_test,
        regression_predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_reg_test,
            regression_predictions,
        )
    )

    r2 = r2_score(
        y_reg_test,
        regression_predictions,
    )

    # Number of observations.
    n = len(y_reg_test)

    # Number of predictors after one-hot encoding.
    transformed_X_reg_test = (
        regression_pipeline
        .named_steps["preprocessor"]
        .transform(X_reg_test)
    )

    p = transformed_X_reg_test.shape[1]

    adj_r2 = adjusted_r2(
        r2,
        n,
        p,
    )

    print(f"MAE:       {mae:.6f}")
    print(f"RMSE:      {rmse:.6f}")
    print(f"R²:        {r2:.6f}")
    print(f"Adjusted R²: {adj_r2:.6f}")

    # Residual plot.
    residuals = y_reg_test - regression_predictions

    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        x=regression_predictions,
        y=residuals,
        alpha=0.65,
    )

    plt.axhline(
        0,
        color="red",
        linestyle="--",
    )

    plt.xlabel("Predicted Fare")
    plt.ylabel("Residual")
    plt.title("Fare Regression Residual Plot")

    plt.tight_layout()
    plt.savefig(
        RESIDUAL_PATH,
        dpi=150,
    )
    plt.close()

    print(f"Saved: {RESIDUAL_PATH}")

    # ========================================================
    # FINAL MODEL COMPARISON TABLE
    # ========================================================

    print_section("FINAL MODEL COMPARISON")

    final_classification = classification_table.copy()

    regression_columns = [
        "MAE",
        "RMSE",
        "R2",
        "Adjusted_R2",
    ]

    for column in regression_columns:
        final_classification[column] = np.nan

    # Add regression metrics as a separate regression row.
    regression_row = {
        "Model": "Linear Regression",
        "Accuracy": np.nan,
        "Precision": np.nan,
        "Recall": np.nan,
        "F1": np.nan,
        "ROC_AUC": np.nan,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Adjusted_R2": adj_r2,
    }

    final_classification = pd.concat(
        [
            final_classification,
            pd.DataFrame([regression_row]),
        ],
        ignore_index=True,
    )

    print(
        final_classification.to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE BEST COMPLETE PIPELINE
    # ========================================================

    print_section("SAVING MODEL ARTIFACT")

    # We use the tuned Random Forest as the final artifact.
    #
    # It contains:
    # - imputation
    # - one-hot encoding
    # - scaling
    # - Random Forest estimator
    #
    # Therefore raw dictionaries/DataFrames can be passed
    # directly to the pipeline at prediction time.

    full_pipeline = best_rf_pipeline

    joblib.dump(
        full_pipeline,
        MODEL_PATH,
    )

    print(f"Saved complete pipeline to: {MODEL_PATH}")

    # ========================================================
    # RELOAD MODEL AND PREDICT ON FAKE RAW DATA
    # ========================================================

    print_section("MODEL RELOAD TEST")

    reloaded_pipeline = joblib.load(
        MODEL_PATH
    )

    fake_raw_data = pd.DataFrame(
        [
            {
                "pclass": 3,
                "sex": "male",
                "age": 30.0,
                "sibsp": 0,
                "parch": 0,
                "fare": 10.0,
                "embarked": "S",
            }
        ]
    )

    fake_prediction = reloaded_pipeline.predict(
        fake_raw_data
    )

    fake_probability = (
        reloaded_pipeline.predict_proba(
            fake_raw_data
        )[0, 1]
    )

    print("Fake raw input:")
    print(fake_raw_data)

    print(
        "Prediction:",
        fake_prediction.tolist(),
    )

    print(
        f"Survival probability: "
        f"{fake_probability:.6f}"
    )

    # ========================================================
    # FINAL OUTPUT SUMMARY
    # ========================================================

    print_section("PIPELINE COMPLETE")

    print("Generated files:")

    files = [
        CSV_PATH,
        UNIVARIATE_PATH,
        CORR_PATH,
        *STORY_PATHS,
        TREE_PATH,
        RESIDUAL_PATH,
        MODEL_PATH,
    ]

    for file in files:
        print(
            f"{'OK' if file.exists() else 'MISSING'} "
            f"- {file.name}"
        )

    # --------------------------------------------------------
    # Return useful objects for programmatic use.
    # --------------------------------------------------------

    return {
        "cleaned": cleaned,
        "classification_results": classification_table,
        "imbalance_results": imbalance_results,
        "grid_search": grid_search,
        "regression_metrics": {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Adjusted_R2": adj_r2,
        },
        "model_path": MODEL_PATH,
    }


if __name__ == "__main__":
    main()
