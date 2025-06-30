# Loan-Approval-Prediction
This project involves building a machine learning model to predict whether a loan application should be approved or rejected based on historical data. It applies various statistical and machine learning techniques to clean, explore, and model the dataset.

## Problem Statement

In the financial industry, automating loan approval processes helps reduce manual effort, increase accuracy, and provide faster decisions. The goal of this project is to predict loan approval using applicant and loan data.

---

## Dataset

- **Rows**: 148,670
- **Columns**: 34
- **Contains**: Demographic details, credit history, loan amount, employment information, and more.
- **Note**: The dataset contains missing values and categorical features that required preprocessing.

---

## TechStack

- **Data Manipulation** : Pandas, Numpy
- **Data Visualization** : Matplotlib, Seaborn
- **Modeling & ML** : Scikit-learn, Statsmodels
- **Model Evaluation** : Scikit-learn Metrics (accuracy, confusion matrix, ROC)
- **Feature Engineering** : StandardScaler, RFE (Recursive Feature Elimination)
- **IDE/Environment** : Jupyter Notebook    

---

## Steps Performed

1. **Data Loading & Initial Exploration**
   - Checked shape, null values, and basic statistics
2. **Data Cleaning**
   - Handled missing values and formatted columns
3. **Exploratory Data Analysis (EDA)**
   - Visualized feature distributions and correlations
4. **Feature Engineering**
   - Encoded categorical variables
   - Scaled numerical values
5. **Model Building**
   - Used Logistic Regression for binary classification
   - Feature selection via RFE (Recursive Feature Elimination)
6. **Model Evaluation**
   - Accuracy Score, Confusion Matrix, ROC Curve.

---

## Model Performance

- **Classifier Used**: Logistic Regression
- **Evaluation Metrics**:
  - Accuracy Score
  - ROC-AUC Curve
  - Confusion Matrix
  - Classification Report

---

## Key Takeaways

- Feature selection played a key role in model performance.
- Some categorical variables like employment type and credit score were strong indicators.
- Proper data scaling improved model generalization.



