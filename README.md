# Healthcare Insurance Cost Prediction using Machine Learning

## Project Overview
This project aims to predict individual healthcare insurance costs using statistical analysis and machine learning models.  
The objective is to identify key factors influencing insurance charges and build an accurate regression model that generalizes well on unseen data.

The project includes data cleaning, exploratory data analysis (EDA), outlier treatment, assumption testing for linear regression, model comparison, hyperparameter tuning, and final model selection.

---

## Dataset Information
- Dataset: Insurance Dataset
- Observations: 1,337
- Features: 6
- Target Variable: `charges` (Insurance cost)

### Feature Description
- Numerical Features: `age`, `bmi`
- Categorical Features: `sex`, `smoker`, `region`
- Target: `charges`

---

## Tools & Technologies Used
- Python  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- Statsmodels  
- Jupyter Notebook / Google Colab  

---

## Data Preprocessing
- Removed duplicate observations
- Standardized categorical labels (e.g., sex column)
- Removed irrelevant feature (`children`)
- Treated BMI outliers using IQR-based clipping
- One-hot encoding for categorical variables
- Feature scaling using StandardScaler
- ColumnTransformer for clean preprocessing pipeline

---

## Exploratory Data Analysis (EDA)
- Distribution analysis of insurance charges
- Gender-wise and smoker-wise cost comparison
- Outlier detection using boxplots
- Correlation analysis using heatmap

### Key EDA Insights
- Smoking status has the strongest impact on insurance charges
- BMI and age are positively correlated with insurance cost
- Insurance charges are right-skewed
- Smokers incur significantly higher medical expenses

---

## Models Implemented
- Linear Regression  
- Decision Tree Regressor  
- Random Forest Regressor  
- Gradient Boosting Regressor  

---

## Model Evaluation Metrics
- R² Score  
- Adjusted R² Score  
- Cross-Validation R²  
- Root Mean Squared Error (RMSE)

---

## Linear Regression Diagnostics
- Linearity assumption violated
- Residuals are independent (Durbin–Watson ≈ 2.07)
- Residuals are not normally distributed

This motivated the use of tree-based ensemble models.

---

## Model Performance Summary

| Model | Test R² | CV R² | RMSE |
|------|--------|------|------|
| Linear Regression | 0.805 | 0.745 | 5984 |
| Random Forest Regressor | 0.900 | 0.851 | 4291 |
| Gradient Boosting Regressor | **0.900** | **0.856** | **4278** |

---

## Final Model Selection
**Gradient Boosting Regressor** was selected as the final model due to:
- Highest cross-validation performance
- Lower RMSE
- Better generalization
- Robust handling of non-linear relationships

### Final Model Performance
- Test R²: **0.901**
- Adjusted R²: **0.899**
- RMSE: **4260**

---

## Feature Importance (Final Model)
- Smoker Status: ~69.6%
- BMI: ~18.0%
- Age: ~12.3%
- Sex: Negligible impact

Smoking status is the dominant predictor of insurance charges.

---

## Prediction Example
The model can predict insurance cost using age, BMI, and smoking status:
```python
predict(age=23, bmi=19.5, smoker="no")
