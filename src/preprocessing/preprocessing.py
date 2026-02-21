import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

dataset = pd.read_csv("../../data/raw/Bank_Customers.csv")

X = dataset.iloc[:, 3:-1]
y = dataset.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build preprocessing
preprocess = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            [
                "CreditScore",
                "Age",
                "Tenure",
                "Balance",
                "NumOfProducts",
                "HasCrCard",
                "IsActiveMember",
                "EstimatedSalary",
            ],
        ),
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            ),
            ["Geography", "Gender"],
        ),
    ]
)

preprocess.fit(X_train)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

joblib.dump(preprocess, os.path.join(MODELS_DIR, "preprocess.pkl"))
