# Zepto Capstone – Titanic Analytics & Machine Learning Pipeline

A complete end-to-end analytics and machine learning pipeline built using the **Seaborn Titanic dataset**.

The project covers data loading, data profiling, missing-value treatment, exploratory data analysis, correlation analysis, visualization, classification, class-imbalance handling, hyperparameter tuning, regression, model persistence, and prediction using a reloaded production-style pipeline.

---

## Project Overview

This project analyzes the Titanic passenger dataset and builds machine learning models to predict passenger survival.

The pipeline performs the following:

1. Loads the Titanic dataset exactly once using Seaborn.
2. Saves the raw dataset as `analytics/titanic.csv`.
3. Profiles the dataset.
4. Analyzes and handles missing values.
5. Performs univariate analysis.
6. Performs bivariate survival analysis.
7. Performs correlation analysis.
8. Creates four data-story visualizations.
9. Performs a standardization sanity check.
10. Builds classification pipelines for:

- Logistic Regression
- Decision Tree
- Random Forest

11. Evaluates classification performance using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- Confusion Matrix

12. Compares class-imbalance strategies:

- Baseline Random Forest
- Class-weighted Random Forest
- SMOTE Random Forest

13. Tunes a Random Forest using `GridSearchCV`.
14. Builds a multivariable Linear Regression model to predict fare.
15. Calculates:

- MAE
- RMSE
- R²
- Adjusted R²

16. Saves the complete tuned classification pipeline using Joblib.
17. Reloads the saved pipeline.
18. Tests the reloaded pipeline using raw input data.
19. Generates all project artifacts inside the `analytics/` directory.

---

### Project Structure

```
zepto_capstone/
│
├── analytics/
│   ├── analytics_pipeline.py
│   ├── titanic.csv
│   ├── univariate.png
│   ├── corr_heatmap.png
│   ├── story_1.png
│   ├── story_2.png
│   ├── story_3.png
│   ├── story_4.png
│   ├── tree.png
│   ├── residuals.png
│   └── model_pipeline.pkl
│
├── venv/
│
├── README.md
└── requirements.txt
```

> Generated files such as `titanic.csv`, PNG visualizations, and `model_pipeline.pkl` are created automatically when the pipeline runs.

---

### Dataset

The project uses the Titanic dataset provided by Seaborn:

```
df = sns.load_dataset("titanic")
```

The dataset contains passenger information such as:

| Feature | Description |
| --- | --- |
| `survived` | Survival indicator |
| `pclass` | Passenger class |
| `sex` | Passenger sex |
| `age` | Passenger age |
| `sibsp` | Number of siblings/spouses aboard |
| `parch` | Number of parents/children aboard |
| `fare` | Passenger fare |
| `embarked` | Port of embarkation |
| `class` | Passenger class as categorical data |
| `who` | Person category |
| `deck` | Deck information |
| `embark_town` | Embarkation town |
| `alive` | Survival as text |
| `alone` | Whether passenger travelled alone |

The raw dataset is loaded only once and immediately saved as:

```
analytics/titanic.csv
```

---

### Data Cleaning

The pipeline follows a missing-value strategy based on the percentage of missing observations.

#### Missing-value rules

| Missing percentage | Treatment |
| --- | --- |
| `< 5%` | Drop affected rows |
| `5% – 30%` | Impute values |
| `> 30%` | Preserve variable and encode missing values |

For the Titanic dataset:

- `age` has approximately 20% missing values and is median-imputed.
- `embarked` has less than 5% missing values, so affected rows are removed.
- `deck` has more than 30% missing values and missing observations are encoded as `"missing"`.

#### Categorical missing-value handling

Because Seaborn's `deck` column is a Pandas categorical column, the `"missing"` category is explicitly added before filling missing values.

```
if isinstance(cleaned[column].dtype, pd.CategoricalDtype):
    if "missing" not in cleaned[column].cat.categories:
        cleaned[column] = (
            cleaned[column]
            .cat.add_categories(["missing"])
        )

cleaned[column] = cleaned[column].fillna("missing")
```

This prevents the Pandas error:

```
TypeError:
Cannot setitem on a Categorical with a new category (missing)
```

---

## Exploratory Data Analysis

### Univariate Analysis

The pipeline analyzes the distributions of:

- Age
- Fare

For each variable, the pipeline calculates IQR-based outliers.

For fare, it also calculates:

- Mean
- Median
- Mode

The resulting visualization is saved as:

```
analytics/univariate.png
```

The visualization contains:

- Age histogram
- Age boxplot
- Fare histogram
- Fare boxplot

---

### Bivariate Analysis

Survival rates are calculated using boolean masking rather than relying exclusively on `groupby()`.

The pipeline calculates survival rates by:

1. Sex
2. Passenger class
3. Sex + passenger class

Example:

```
mask = df["sex"] == value
rate = df.loc[mask, "survived"].mean()
```

This provides a direct comparison of survival outcomes across passenger groups.

---

## Correlation Analysis

The following numeric variables are analyzed:

```
survived
pclass
age
sibsp
parch
fare
```

A Pearson correlation matrix is calculated.

The resulting heatmap is saved as:

```
analytics/corr_heatmap.png
```

The pipeline also identifies the two strongest absolute correlations among the selected numeric variables.

---

## Data Story Visualizations

Four visualizations are generated.

### Story 1 – Survival by Sex

Shows survival rates for male and female passengers.

Output:

```
analytics/story_1.png
```

### Story 2 – Survival by Passenger Class

Shows survival rates across passenger classes.

Output:

```
analytics/story_2.png
```

### Story 3 – Survival by Sex and Passenger Class

Combines sex and passenger class to provide a more granular survival comparison.

Output:

```
analytics/story_3.png
```

### Story 4 – Age, Fare and Survival

A scatter plot showing:

- Age
- Fare
- Survival status

Output:

```
analytics/story_4.png
```

---

## Standardization Sanity Check

The pipeline performs a manual standardization check on:

- Age
- Fare

The transformation is:

```
z = (x - mean) / standard_deviation
```

The script prints the mean and standard deviation before and after standardization.

Expected behavior after standardization:

```
Mean ≈ 0
Standard deviation ≈ 1
```

This provides a sanity check that the standardization calculation is functioning correctly.

---

## Classification

The target variable is:

```
survived
```

The classification features are:

```
pclass
sex
age
sibsp
parch
fare
embarked
```

The data is divided into training and testing sets using a stratified split:

```
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)
```

Stratification preserves the approximate class distribution between the training and testing datasets.

---

## Preprocessing Pipeline

Numerical variables are processed using:

```
Median Imputation
        ↓
StandardScaler
```

Categorical variables are processed using:

```
Most-Frequent Imputation
        ↓
OneHotEncoder
```

The complete preprocessing structure is implemented using `ColumnTransformer`.

This prevents preprocessing from being performed manually outside the machine learning pipeline.

---

## Classification Models

Three classification algorithms are evaluated.

### Logistic Regression

```
LogisticRegression(
    max_iter=2000,
    random_state=42
)
```

### Decision Tree

```
DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)
```

### Random Forest

```
RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
```

Each model is combined with the preprocessing pipeline.

---

## Classification Evaluation

The models are evaluated using:

### Accuracy

Percentage of correctly classified observations.

#### Precision

Measures how many predicted positive cases were actually positive.

#### Recall

Measures how many actual positive cases were correctly identified.

#### F1 Score

Harmonic mean of precision and recall.

#### ROC AUC

Measures discrimination between the two target classes based on predicted probabilities.

#### Confusion Matrix

Displays:

```
True Negative
False Positive
False Negative
True Positive
```

---

## Decision Tree Visualization

The fitted Decision Tree is visualized using Scikit-learn's `plot_tree()`.

The tree visualization is saved as:

```
analytics/tree.png
```

The pipeline obtains the transformed feature names from the fitted `ColumnTransformer`.

---

## Class Imbalance Analysis

Three Random Forest strategies are compared.

### 1\. Baseline Random Forest

Uses the standard Random Forest configuration.

### 2\. Class-Weighted Random Forest

Uses:

```
class_weight="balanced"
```

This adjusts the model's treatment of the target classes based on their frequencies.

### 3\. SMOTE Random Forest

Uses Synthetic Minority Oversampling Technique.

The pipeline is:

```
Preprocessor
     ↓
SMOTE
     ↓
Random Forest
```

SMOTE is placed inside the `imblearn` pipeline so that oversampling is performed as part of the model-training process rather than manually before splitting the data.

The three approaches are compared using:

- Precision
- Recall
- F1 Score

---

## Random Forest Hyperparameter Tuning

A `GridSearchCV` search is performed for the Random Forest.

The parameter grid contains:

```
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__max_features": ["sqrt", "log2"],
}
```

This produces a systematic search across different Random Forest configurations.

The model is evaluated using:

```
scoring="f1"
```

and:

```
cv=5
```

The best estimator is refitted automatically.

The tuned Random Forest is then evaluated on the held-out test set.

The pipeline also reports the Random Forest out-of-bag score.

---

## Fare Regression

A separate regression model predicts:

```
fare
```

using:

```
pclass
sex
age
sibsp
parch
embarked
```

The model is:

```
LinearRegression()
```

Categorical variables are one-hot encoded, while numeric variables are median-imputed.

---

## Regression Evaluation

The regression model is evaluated using four metrics.

### MAE

Mean Absolute Error measures the average absolute difference between predicted and actual fares.

### RMSE

Root Mean Squared Error gives greater weight to larger prediction errors.

### R²

R-squared measures the proportion of variance explained by the regression model.

### Adjusted R²

Adjusted R-squared accounts for the number of predictors in the model.

The formula used is:

```
Adjusted R² =
1 - ((1 - R²) × (n - 1) / (n - p - 1))
```

where:

- `n` = number of observations
- `p` = number of predictors

---

## Regression Residual Analysis

Residuals are calculated as:

```
Residual = Actual Fare - Predicted Fare
```

A residual scatter plot is generated to examine the relationship between predicted values and residual errors.

Output:

```
analytics/residuals.png
```

---

## Model Persistence

The tuned Random Forest pipeline is saved using Joblib:

```
joblib.dump(
    full_pipeline,
    MODEL_PATH,
)
```

Output:

```
analytics/model_pipeline.pkl
```

The saved artifact contains the complete preprocessing and machine learning pipeline, including:

```
Raw input
    ↓
Missing-value imputation
    ↓
Scaling
    ↓
One-hot encoding
    ↓
Tuned Random Forest
    ↓
Prediction
```

This allows raw DataFrames to be passed directly to the saved model.

---

## Model Reload Test

After saving the model, the pipeline is reloaded:

```
reloaded_pipeline = joblib.load(
    MODEL_PATH
)
```

A raw passenger record is then supplied:

```
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
```

The reloaded model produces:

- Survival prediction
- Survival probability

This confirms that the persisted pipeline can be used independently after being reloaded.

---

## Installation

### 1\. Clone or download the project

Place the project in a local directory.

Example:

```
zepto_capstone/
```

### 2\. Create a virtual environment

Windows:

```
python -m venv venv
```

Activate it:

```
venv\Scripts\activate
```

Linux/macOS:

```
python3 -m venv venv
source venv/bin/activate
```

### 3\. Install dependencies

Install the required packages:

```
pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn joblib
```

Alternatively, create a `requirements.txt` file containing:

```
pandas
numpy
seaborn
matplotlib
scikit-learn
imbalanced-learn
joblib
```

Then run:

```
pip install -r requirements.txt
```

---

## Running the Pipeline

From the project directory:

```
python analytics/analytics_pipeline.py
```

The script will:

1. Load the Titanic dataset.
2. Save the raw CSV.
3. Clean the data.
4. Generate analysis output.
5. Train classification models.
6. Compare imbalance strategies.
7. Tune Random Forest.
8. Train the regression model.
9. Generate visualizations.
10. Save the final model.
11. Reload the model.
12. Run a prediction test.

---

## Expected Output

During execution, the terminal displays sections such as:

```
================================================================================
PART A - DATA LOADING
================================================================================
```

followed by:

```
DATA INFO
DESCRIPTIVE STATISTICS
DATA SHAPE
MISSING VALUE ANALYSIS
CLEANING
UNIVARIATE ANALYSIS
BIVARIATE SURVIVAL ANALYSIS
CORRELATION ANALYSIS
DATA STORY
STANDARDIZATION CHECK
PART B - PREDICTIVE MODELING
```

The pipeline subsequently reports classification metrics, imbalance comparisons, Random Forest tuning results, regression metrics, and model persistence results.

---

## Generated Artifacts

After a successful run, the `analytics/` directory contains:

| File | Purpose |
| --- | --- |
| `titanic.csv` | Raw Titanic dataset |
| `univariate.png` | Age and fare distribution analysis |
| `corr_heatmap.png` | Numeric correlation heatmap |
| `story_1.png` | Survival by sex |
| `story_2.png` | Survival by passenger class |
| `story_3.png` | Survival by sex and class |
| `story_4.png` | Age, fare and survival |
| `tree.png` | Decision Tree visualization |
| `residuals.png` | Fare regression residual plot |
| `model_pipeline.pkl` | Persisted tuned Random Forest pipeline |

---

## Technologies Used

- **Python**
- **Pandas** – data manipulation and analysis
- **NumPy** – numerical computation
- **Seaborn** – dataset loading and visualization
- **Matplotlib** – visualization
- **Scikit-learn** – preprocessing, modeling and evaluation
- **imbalanced-learn** – SMOTE
- **Joblib** – model persistence

---

## Machine Learning Workflow

The overall workflow can be summarized as:

```
Titanic Dataset
       │
       ▼
Load with Seaborn
       │
       ▼
Save Raw CSV
       │
       ▼
Data Profiling
       │
       ▼
Missing-Value Analysis
       │
       ▼
Data Cleaning
       │
       ├───────────────┐
       ▼               ▼
Exploratory Analysis  Modeling
       │               │
       ├── Univariate  ├── Classification
       ├── Bivariate   ├── Imbalance Analysis
       ├── Correlation ├── Grid Search
       └── Data Story  └── Regression
                       │
                       ▼
                 Model Evaluation
                       │
                       ▼
                Save Final Pipeline
                       │
                       ▼
                 Reload Pipeline
                       │
                       ▼
                   Prediction
```

---

## Reproducibility

The project uses fixed random seeds in the machine learning workflow, including:

```
random_state=42
```

This is used for:

- Train/test splitting
- Decision Tree
- Random Forest
- SMOTE
- Random Forest grid search

This improves reproducibility across executions, subject to differences in library versions and execution environments.

---

## Important Design Decisions

### Raw data is loaded exactly once

The dataset is loaded only through:

```
sns.load_dataset("titanic")
```

The resulting DataFrame is immediately saved to CSV.

#### Data leakage is minimized

Preprocessing for machine learning is performed inside Scikit-learn pipelines.

This ensures transformations such as imputation, scaling, and encoding are fitted as part of the training process.

#### SMOTE is inside the modeling pipeline

SMOTE is not applied to the entire dataset before train/test splitting.

Instead:

```
Training data
     ↓
Preprocessing
     ↓
SMOTE
     ↓
Random Forest
```

This avoids contaminating the test set with synthetic observations.

#### Redundant target variables are excluded

The classification model predicts `survived`, so directly duplicative variables such as `alive` are not included.

---

## Troubleshooting

### `Cannot setitem on a Categorical`

If you see:

```
TypeError:
Cannot setitem on a Categorical with a new category (missing)
```

the issue occurs when assigning `"missing"` to a Pandas categorical column.

Add the category first:

```
if "missing" not in cleaned[column].cat.categories:
    cleaned[column] = (
        cleaned[column]
        .cat.add_categories(["missing"])
    )

cleaned[column] = cleaned[column].fillna("missing")
```

---

### `sparse_output` is not recognized

If your Scikit-learn version does not support:

```
sparse_output=False
```

use:

```
sparse=False
```

instead.

The `sparse_output` parameter is used by newer Scikit-learn versions.

---

### SMOTE is not installed

Install `imbalanced-learn`:

```
pip install imbalanced-learn
```

---

### Model file not found

Run the complete pipeline first:

```
python analytics/analytics_pipeline.py
```

The model is generated automatically at:

```
analytics/model_pipeline.pkl
```

---

## Conclusion

This project demonstrates a complete analytics and machine learning workflow using the Titanic dataset.

It combines:

- Data engineering
- Data cleaning
- Exploratory data analysis
- Statistical analysis
- Data visualization
- Classification
- Imbalanced-learning techniques
- Hyperparameter tuning
- Regression
- Model evaluation
- Model serialization
- Model reloading
- Prediction on raw data

The final output is a reusable machine learning pipeline that combines preprocessing and the tuned Random Forest model into a single persisted artifact.
