# Bank Customer Churn Prediction

This repository contains the **final project for the Artificial Intelligence course** at **K.N. Toosi University of Technology**.

The goal of the project is to build a **binary classification** model that predicts whether a bank customer will **exit (churn)** based on their profile and account-related features. The workflow includes **data analysis (EDA)**, **data preprocessing**, training a **neural network model**, and evaluating performance using classification metrics.

## Project Overview
- **Task:** Customer churn prediction (binary classification)
- **Input:** Tabular customer data (numerical + categorical features)
- **Output:** `Exited` (0 = not churned, 1 = churned)

## Repository Structure

```text
.
├── data/
|   ├── raw/
|   ├── processed/
├── notebooks/
|   ├── EDA.ipynb
|   ├── experiments/
├── src/
|   ├── preprocessing/
|   ├── models/
|   ├── training/
|   ├── evaluation/
├── reports/
|   ├── charts/
|   ├── metrics/
├── requirements.txt
└── README.md
```

## Methodology
1. **Exploratory Data Analysis (EDA)**
   - Visualize distributions, churn rates, and important relationships in the data.
2. **Preprocessing**
   - Handle missing values (imputation)
   - Encode categorical variables (one-hot encoding)
   - Scale numerical features (standardization)
3. **Modeling**
   - Train an Artificial Neural Network (TensorFlow/Keras) as a baseline model
   - Improve the model using regularization/tuning
4. **Evaluation**
   - Accuracy
   - Precision / Recall / F1-score
   - ROC-AUC
   - Confusion Matrix

## Setup & Usage

### 1) Install dependencies
```bash
pip install -r requirements.txt
```
### 2) Run the notebooks
Open the notebook(s) inside the notebooks/ folder.