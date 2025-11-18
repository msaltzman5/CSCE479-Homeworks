# main.py
import tensorflow as tf
from model import Model
import util


def run(
    name,
    arch_config,
    hyperparams,
    train_ds,
    val_ds,
    test_ds,
    vocab_size,
):
    print(f"\n=== {name} ===")
    print(f"Architecture: {arch_config}")
    print(f"Hyperparams:  {hyperparams}")

    model = Model(
        vocab_size=vocab_size,
        embedding_dim=hyperparams["embedding_dim"],
        rnn_units=hyperparams["rnn_units"],
        dense_units=hyperparams["dense_units"],
        rnn_type=arch_config["rnn_type"],
        bidirectional=arch_config["bidirectional"],
        dropout_rate=hyperparams["dropout"],
        l2_reg=hyperparams["l2"],
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparams["learning_rate"]),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,              # required minimum patience
        restore_best_weights=True,
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=hyperparams["max_epochs"],
        verbose=2,
        callbacks=[early_stopping],
    )

    val_metrics = model.evaluate(val_ds, return_dict=True, verbose=0)
    test_metrics = model.evaluate(test_ds, return_dict=True, verbose=0)

    print(f"VAL  -> loss={val_metrics['loss']:.4f}  acc={val_metrics['accuracy']:.4f}")
    print(f"TEST -> loss={test_metrics['loss']:.4f} acc={test_metrics['accuracy']:.4f}")

    # Confusion matrix plot
    cm_filename = f"{name}_confusion_matrix.png"
    util.confusion_matrix_plot(model, test_ds, filename=cm_filename)
    print(f"Saved confusion matrix to {cm_filename}")

    # Confidence intervals
    n_test = sum(1 for _ in test_ds.unbatch())
    ci_low, ci_high = util.confidence_interval(test_metrics["accuracy"], n_test)
    print(f"95% CI for test accuracy: [{ci_low:.4f}, {ci_high:.4f}]")


def main():
    # Load datasets + vocabulary info
    train_ds, val_ds, test_ds, vocab_size, seq_len = util.parse_dataset()

    print(f"Vocab size: {vocab_size}, sequence length: {seq_len}")

    # Two different sequential architectures
    arches = {
        "BiLSTM": {
            "rnn_type": "lstm",
            "bidirectional": True,
        },
        "BiGRU": {
            "rnn_type": "gru",
            "bidirectional": True,
        },
    }

    # Two different hyperparameter settings (per architecture)
    hyperparam_settings = [
        {
            "embedding_dim": 64,
            "rnn_units": 64,
            "dense_units": 32,
            "dropout": 0.3,
            "l2": 1e-4,
            "learning_rate": 1e-3,
            "max_epochs": 20,
        },
        {
            "embedding_dim": 128,
            "rnn_units": 128,
            "dense_units": 64,
            "dropout": 0.5,
            "l2": 1e-5,
            "learning_rate": 5e-4,
            "max_epochs": 25,
        },
    ]

    # This yields 2 architectures × 2 hyperparameter settings = 4 models
    for arch_name, arch_config in arches.items():
        for i, hp in enumerate(hyperparam_settings, start=1):
            name = f"{arch_name}_cfg{i}"
            run(
                name=name,
                arch_config=arch_config,
                hyperparams=hp,
                train_ds=train_ds,
                val_ds=val_ds,
                test_ds=test_ds,
                vocab_size=vocab_size,
            )


if __name__ == "__main__":
    main()