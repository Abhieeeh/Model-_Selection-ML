# Model Selection and EDA for Car Price Prediction

This repository contains a comprehensive Machine Learning pipeline for predicting car prices using the **CarPrice** dataset. The project walks through Exploratory Data Analysis (EDA), categorical encoding techniques, model comparison, and hyperparameter optimization to select the best-performing regression model.

## 📋 Project Overview

The primary goal of this project is to analyze the factors affecting car prices and build an accurate regression model to predict the price of a car based on its features.

### Workflow & Features
1. **Exploratory Data Analysis (EDA)**:
   - Descriptive statistics and data structure inspection.
   - Feature relationship analysis using statistical tests:
     - **Pearson Correlation** for numeric relationships.
     - **ANOVA (Analysis of Variance)** (`f_oneway`) and **Independent t-tests** (`ttest_ind`) to verify statistical significance of categorical predictors.
2. **Data Preprocessing**:
   - Feature selection based on correlation and statistical significance.
   - Categorical encoding methods: Label Encoding, One-Hot Encoding, and Target Encoding.
3. **Model Evaluation & Selection**:
   - Training multiple regressor models.
   - Evaluating using Mean Absolute Error (MAE), Mean Squared Error (MSE), and Coefficient of Determination ($R^2$ Score).
4. **Hyperparameter Tuning**:
   - Grid search and randomized search (`RandomizedSearchCV`) to optimize the best candidate model.

---

## 📊 Model Performance Comparison

Here is the evaluation of the baseline models before hyperparameter tuning:

| Model | MAE | MSE | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Random Forest Regressor** | **1,279.95** | **3.355 × 10⁶** | **0.9575** |
| Gradient Boosting Regressor | 1,660.49 | 5.878 × 10⁶ | 0.9255 |
| Decision Tree Regressor | 1,754.76 | 6.454 × 10⁶ | 0.9182 |
| Linear Regression | 2,437.27 | 1.507 × 10⁷ | 0.8091 |
| SVR (Support Vector Regressor) | 5,707.55 | 8.702 × 10⁷ | -0.1023 |

### Key Finding
* **Random Forest Regressor** achieved the highest baseline performance with an $R^2$ score of **~95.75%**.

---

## ⚙️ Hyperparameter Tuning

`RandomizedSearchCV` was applied to the `RandomForestRegressor` to tune parameters across the following search space:
- `n_estimators`: `[100, 200, 300, 500]`
- `max_depth`: `[None, 10, 20, 30, 40]`
- `min_samples_split`: `[2, 5, 10]`
- `min_samples_leaf`: `[1, 2, 4]`
- `max_features`: `['sqrt', 'log2', None]`

### Best Parameters Found:
```python
{
    'n_estimators': 300,
    'max_depth': None,
    'max_features': 'sqrt',
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42
}
```

The optimized model generalized extremely well with a final validation $R^2$ score of **~0.9314** and MAE of **~1,398.89**.

---

## 🛠️ Tech Stack & Requirements

This project is implemented in Python. The key libraries used are:

* **Data Wrangling**: `pandas`, `numpy`
* **Statistical Analysis**: `scipy` (Pearson correlation, ANOVA, t-tests)
* **Machine Learning**: `scikit-learn` (regressors, preprocessors, metrics, cross-validation)
* **Environment**: Jupyter Notebook / Google Colab

---

## 🚀 How to Run the Notebook

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Abhieeeh/Model-_Selection-ML.git
   cd Model-_Selection-ML
   ```

2. **Install dependencies**:
   ```bash
   pip install numpy pandas scipy scikit-learn notebook
   ```

3. **Launch the Notebook**:
   ```bash
   jupyter notebook Model_Selection.ipynb
   ```