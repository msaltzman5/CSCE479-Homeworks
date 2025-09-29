import tensorflow as tf
from model import Model
import util

def main():
    train_ds, val_ds, test_ds = util.parse_dataset()

    # Outline: 
    # 2 different fully-connected architectures
    #   - At least 1 hidden layer
    #   - Hidden layers use ReLU
    #   - Output layers uses softmax
    # Adam as the optimizer
    # 2 different hyperparameters: (i.e., learning rates, batch size)
    # 2 different reglularizations (i.e., dropout, L2 regularization)

    # Testing model
    model = Model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"]
    )

    # Train
    model.fit(train_ds, validation_data=val_ds, epochs=10)

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    main()