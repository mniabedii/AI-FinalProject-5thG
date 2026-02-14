import numpy as np
import tensorflow as tf
import wandb

from sklearn.utils.class_weight import compute_class_weight
from wandb.integration.keras import WandbMetricsLogger

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(6, activation="relu", input_shape=(X_train.shape[1],)))

model.add(tf.keras.layers.Dense(6, activation="relu"))

model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

class_weights = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
class_weight_dict = dict(zip(np.unique(y_tr), class_weights))

print("Class Weights:", class_weight_dict)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_model.keras",
    monitor='val_loss',
    save_best_only=True,
)

RUN_NAME = "model_2_exp_4"

wandb.init(
    entity="arminfakhar1384-",
    project="bank_customers_churn",
    name=RUN_NAME,
    config={
        "model_id": 24,
        "architecture": "6-6-relu + sigmoid",
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "loss": "binary_crossentropy",
        "class_weight": "balanced"
    }
)

wandb_callback = WandbMetricsLogger()

history = model.fit(
    X_tr,
    y_tr,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[checkpoint, wandb_callback],
    verbose=1
)




