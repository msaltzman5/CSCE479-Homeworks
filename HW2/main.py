import tensorflow as tf
from model import ModelA, ModelB
import util

# Function to train & evaluate a given model
def run_model(model, name, train_ds, val_ds, test_ds, learning_rate=1e-3, epochs=50, patience=5):
    print(f"\n=== Training {name} ===")
    print(f"Learning rate={learning_rate}  Dropout={model.dropout}  L2={model.reg.l2 if model.reg else 0}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    # lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    #     monitor='val_loss',
    #     factor=0.5,
    #     patience=4,
    #     min_lr=1e-6,
    #     verbose=1
    # )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=2,
        callbacks=[early_stopping]
        # callbacks=[early_stop, lr_scheduler]
    )

    val_metrics = model.evaluate(val_ds, return_dict=True, verbose=0)
    test_metrics = model.evaluate(test_ds, return_dict=True, verbose=0)

    print(f"VAL  -> loss={val_metrics['loss']:.4f}  acc={val_metrics['accuracy']:.4f}")
    print(f"TEST -> loss={test_metrics['loss']:.4f} acc={test_metrics['accuracy']:.4f}")
    return val_metrics, test_metrics


def main():
    train_ds, val_ds, test_ds = util.parse_dataset()

    architectures = {
        "ModelA": ModelA(num_classes=100, dropout=0.3, l2=5e-5),
        "ModelB": ModelB(num_classes=100, dropout=0.4, l2=1e-4)
    }

    hyperparams = [
        {"learning_rate": 1e-4, "epochs": 30},
        {"learning_rate": 5e-4, "epochs": 40}
    ]

    for name, model in architectures.items():
        for params in hyperparams:
            run_name = f"{name}_lr{params['learning_rate']}_ep{params['epochs']}"
            run_model(
                model=model,
                name=run_name,
                train_ds=train_ds,
                val_ds=val_ds,
                test_ds=test_ds,
                learning_rate=params["learning_rate"],
                epochs=params["epochs"],
                patience=10
            )

if __name__ == '__main__':
    main()