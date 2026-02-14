import numpy as np

from sklearn.metrics import classification_report, roc_curve, confusion_matrix
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

val_probs = model.predict(X_val).ravel()

thresholds = np.linspace(0.01, 0.99, 99)
f1s, precs, recs = [], [], []

for t in thresholds:
    val_pred = (val_probs >= t).astype(int)
    f1s.append(f1_score(y_val, val_pred))                
    precs.append(precision_score(y_val, val_pred, zero_division=0))
    recs.append(recall_score(y_val, val_pred, zero_division=0))

best_idx = int(np.argmax(f1s))
best_t = float(thresholds[best_idx])

print("Best threshold (val):", best_t)
print("Val F1:", f1s[best_idx])
print("Val Precision:", precs[best_idx])
print("Val Recall:", recs[best_idx])

test_probs = model.predict(X_test).ravel()
test_pred = (test_probs >= best_t).astype(int)

print("Using threshold:", best_t)
print("Confusion Matrix:\n", confusion_matrix(y_test, test_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, test_pred, digits=4))

print("Test ROC-AUC:", roc_auc_score(y_test, test_probs))

fpr, tpr, thresholds = roc_curve(y_test, test_probs)
roc_auc = roc_auc_score(y_test, test_probs)


