import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE


from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve


# building model-1

model = tf.keras.models.Sequential()
tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],))
model.add(tf.keras.layers.Dense(units=32, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# smote

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print("Before SMOTE:", np.bincount(y_tr))

smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy=0.6)
X_tr_sm, y_tr_sm = smote.fit_resample(X_tr, y_tr)

print("After  SMOTE:", np.bincount(y_tr_sm))

# compiling the model

tf.keras.backend.clear_session()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

# saving checkpoint

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_model.keras",
    monitor='val_loss',
    save_best_only=True,
)

# Experiment Tracking Configuration (WandB)

RUN_NAME = "model_1_exp_2"

wandb.init(
    project="bank_customers_churn",
    name=RUN_NAME,
    config={
        "model_id": 1,
        "architecture": "64-32-relu + sigmoid",
        "epochs": 50,
        "batch_size": 128,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "loss": "binary_crossentropy"
    }
)

wandb_callback = WandbMetricsLogger()

# training the ANN

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=128,
    callbacks=[checkpoint, wandb_callback]
)