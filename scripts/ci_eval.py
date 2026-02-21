import json
from pathlib import Path
import tensorflow as tf

def main():
    x = tf.random.normal([128, 16])
    y = tf.random.uniform([128], maxval=2, dtype=tf.int32)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(2)
    ])
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    loss, acc = model.evaluate(x, y, verbose=0)

    out_dir = Path("reports/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ci_metrics.json").write_text(json.dumps({"loss": float(loss), "accuracy": float(acc)}, indent=2))
    print("EVAL OK -> reports/metrics/ci_metrics.json")

if __name__ == "__main__":
    main()
