import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix

# ===== 1) find best threshold on VAL =====
y_val_prob = model.predict(X_val).ravel()

thresholds = np.linspace(0.05, 0.95, 181)
min_recall_pos = 0.6

best_t = None
best_macro_f1 = -1
best_stats = None

for t in thresholds:
    y_val_pred = (y_val_prob >= t).astype(int)

    # macro F1
    _, _, f1_macro, _ = precision_recall_fscore_support(
        y_val, y_val_pred, average="macro", zero_division=0
    )

    # positive-class precision/recall/f1
    p1, r1, f11, _ = precision_recall_fscore_support(
        y_val, y_val_pred, average="binary", zero_division=0
    )

    if r1 >= min_recall_pos and f1_macro > best_macro_f1:
        best_macro_f1 = f1_macro
        best_t = t
        best_stats = (p1, r1, f11)

print("Best threshold:", best_t)
print("VAL macro F1:", best_macro_f1)
print("VAL pos P/R/F1:", best_stats)

# ===== 2) evaluate on TEST using that threshold =====
y_test_prob = model.predict(X_test).ravel()
y_test_pred = (y_test_prob >= best_t).astype(int)

print("\nTEST report @ threshold =", best_t)

# confusion matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

# Performance metrics
results = model.evaluate(X_test, y_test, verbose=0)
print("Test Results (loss, acc, precision, recall, auc):\n", results)

print("\nClassification Report:\n")
print(classification_report(y_test, y_test_pred, digits=4))

roc_auc = roc_auc_score(y_test, y_test_prob)
print("ROC-AUC:", roc_auc)

#Plotting ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f'ANN (AUC = {roc_auc:.3f})')
plt.plot([0,1], [0,1], linestyle='--')  # random classifier line

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()