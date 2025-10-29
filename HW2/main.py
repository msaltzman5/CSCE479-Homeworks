import tensorflow as tf
from util import load_cifar100
from model import cnn_model_1, cnn_model_2

def train_and_evaluate(model_fn, learning_rate, batch_size, l2, dropout, max_epochs=50, patience=5):
    train_ds, val_ds, test_ds, num_classes = load_cifar100(batch_size=batch_size)
    model = model_fn(num_classes=num_classes, l2=l2, dropout=dropout)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )

    print(f"\nTraining {model.name} with lr={learning_rate}, bs={batch_size}, l2={l2}, dropout={dropout}")
    history = model.fit(train_ds, validation_data=val_ds, epochs=max_epochs, callbacks=[early_stop], verbose=2)

    print("Evaluating on test set...")
    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"Test Accuracy: {acc:.4f}, Loss: {loss:.4f}\n")

def main():
    # Two models × two hyperparameter sets = 4 runs total
    configs = [
        (cnn_model_1, 1e-3, 128, 1e-4, 0.3),
        (cnn_model_1, 3e-4, 256, 5e-4, 0.4),
        (cnn_model_2, 1e-3, 128, 5e-4, 0.5),
        (cnn_model_2, 5e-4, 256, 1e-3, 0.5),
    ]
    for model_fn, lr, bs, l2, do in configs:
        train_and_evaluate(model_fn, lr, bs, l2, do)

if __name__ == "__main__":
    main()
