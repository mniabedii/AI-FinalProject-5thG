# Bank Customer Churn Prediction

Final project for the Artificial Intelligence course at **K.N. Toosi University of Technology**.

The goal of the project is to build a **binary classification** model that predicts whether a bank customer will **exit (churn)** based on their profile and account-related features. The workflow includes **data analysis (EDA)**, **data preprocessing**, training a **neural network model**, and evaluating performance using classification metrics.

## Project Overview
- **Task:** Customer churn prediction (binary classification)
- **Input:** Tabular customer data (numerical + categorical features)
- **Output:** `Exited` (0 = not churned, 1 = churned)

## Problem
Customer churn prediction is an important business problem.
The goal is to detect customers who are likely to leave the bank using their profile and account-related features.

This is a supervised learning task on structured tabular data.

Since churn prediction is sensitive to missing actual churners, we focused on improving the Precision–Recall balance.

We observed the expected tradeoff between precision and recall and tuned the model accordingly.

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


## Project Structure
```
.
├── data/
|   └── raw/Bank_Customers.csv
|
├── notebooks/
|   ├── EDA.ipynb
|   └── experiments/
|    
├── src/
|   ├── preprocessing/
|   ├── training/
|   ├── evaluation/
|   └── models/
|
├── reports/
|   ├── charts/    
|   └── metrics/
|
├── requirements.txt
└── README.md
````
### Folder Description
- **data/**

    Contains the dataet used for training and evaluation

- **notebooks/**
    
    Includes:
    - EDA notebook
    - All experimental notebooks
    - Model comparison and analysis

- **src/**
    
    Modular Python implementation of the pipeline:
    
    Stores:
    - Training curves
    - Confusion matrices
    - Metric outputs
    - Comparison charts

## Setup & Usage

### 1. Install dependencies:
```
pip install -r requirements.txt
```
### 2. Run notebooks inside notebooks/ or use training scripts inside src/

## Technologies
- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas / NumPy
- Matplotlib / WandB
