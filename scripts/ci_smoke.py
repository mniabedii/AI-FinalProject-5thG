import tensorflow as tf

def main():
    x = tf.random.normal([256, 16])
    y = tf.random.uniform([256], maxval=2, dtype=tf.int32)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(2)
    ])
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(x, y, epochs=1, batch_size=32, verbose=0)
    print("SMOKE TRAIN OK")

if __name__ == "__main__":
    main()
