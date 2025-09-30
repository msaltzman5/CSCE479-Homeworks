import tensorflow as tf
from model import Model
import util

def run(name, hidden_units, learning_rate, dropout, l2, train_ds, val_ds, test_ds):
    print(f"\n=== {name} ===")
    print(f"arch={hidden_units} lr={learning_rate} dropout={dropout} l2={l2} epochs={10}")

    model = Model(hidden_units=hidden_units, learning_rate=learning_rate, dropout=dropout, l2=l2)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"]
    )

    model.fit(train_ds, validation_data=val_ds, epochs=10, verbose=2)

    val_metrics  = model.evaluate(val_ds,  return_dict=True, verbose=0)
    test_metrics = model.evaluate(test_ds, return_dict=True, verbose=0)

    print(f"VAL  -> loss={val_metrics['loss']:.4f}  acc={val_metrics['accuracy']:.4f}")
    print(f"TEST -> loss={test_metrics['loss']:.4f} acc={test_metrics['accuracy']:.4f}")

    # Minimal confusion matrices (numbers only, no plots)
    test_conf_matrix = util.confusion_matrix(model, test_ds).numpy()
    print("Test Confusion Matrix (rows=true, cols=pred):")
    print(test_conf_matrix)


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

    # NEXT TODO: implement another model with different hyperparameters

    # Setting up our variables:
    arches = {
        "A": [256, 128],
        "B": [512, 256, 128],
    }

    # Two learning rates
    learning_rates = [1e-3, 3e-4]

    # Regularization off vs on (dropout + L2 together)
    regs = [
        {"dropout": 0.0, "l2": 0.0},
        {"dropout": 0.2, "l2": 1e-4},
    ]

    for arch_name, hidden_units in arches.items():
        for learning_rate in learning_rates:
            for reg in regs:
                name = f"{arch_name}_lr_{learning_rate}_drop_{reg['dropout']}_l2_{reg['l2']}"
                run(name, hidden_units, learning_rate, reg['dropout'], reg['l2'], train_ds, val_ds, test_ds)

    # Model 1
    # model1 = Model()

    # model1.compile(
    #     optimizer=tf.keras.optimizers.Adam(),
    #     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    #     metrics=["accuracy"]
    # )

    # # Train
    # model1.fit(train_ds, validation_data=val_ds, epochs=10)

    # # Evaluate on test set
    # test_loss, test_acc = model1.evaluate(test_ds)
    # print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    main()