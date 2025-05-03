# Customer_Churn_Prediction
Project Objective
The goal of this project is to predict whether a customer will churn (i.e., stop using the
service) using machine learning. We use the Telco Customer Churn dataset and apply
Logistic Regression and Random Forest classifiers to make predictions.
Importing Required Libraries

# Dataset
 Name: WA_Fn-UseC_-Telco-Customer-Churn.csv
 Source: IBM Telco Dataset Kaggle
 Records: ~7,043 rows
 Target variable: Churn (Yes or No)
 Features include: tenure, contract type, payment method, charges, services used, etc.
Load the Dataset

# Data Cleaning
Operations Performed:
1. Dropped irrelevant columns:
customerID is just an identifier and not useful for prediction.
2. Handled TotalCharges column:
Some values were blank (&#39; &#39;) → replaced with NaN
Converted the column to numeric (float).
Removed rows with missing TotalCharges data.
Purpose:
Ensure the dataset is clean, complete, and numerically consistent for ML algorithms.

# Exploratory Data Analysis (EDA)
EDA helps in understanding the data and identifying trends.
Key Visualizations:
1. Churn Distribution: Shows overall churn ratio (imbalanced dataset).
2. Churn vs Contract Type: Reveals that month-to-month contracts have higher
churn.
3. Churn vs Tenure: Customers with longer tenure churn less.
4. Correlation Heatmap: Shows how features like MonthlyCharges and TotalCharges
are related.
Purpose:
Helps in understanding customer behavior and feature relationships.
Informs modeling strategy and feature importance.

# Data Preprocessing
Encoding:
All categorical (textual) columns (e.g., Yes/No, Male/Female) were converted to numbers
using LabelEncoder.
Feature Scaling:
Used StandardScaler to scale all features.
Standardization helps algorithms like Logistic Regression perform better.
Purpose:
Converts the data into a machine-readable format.
Scaling ensures that no feature dominates due to its scale.

# Model Building
Model 1: Logistic Regression
A simple and interpretable classification model.
Used with max_iter=1000 for convergence.
Trained using X_train, predicted on X_test.

Model 2: Random Forest
An ensemble learning method using multiple decision trees.
Better at handling non-linear patterns.
Trained on the same dataset.

# Model Evaluation
Plot confusion matrix for Logistic Regression
Plot confusion matrix for Random Forest
