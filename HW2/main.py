import tensorflow as tf
from model import ModelA, ModelB
import util

def run_model(model, name, train_ds, val_ds, test_ds, learning_rate=1e-3, epochs=10):
    print(f"\n=== Training {name} ===")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=2)
    val_metrics = model.evaluate(val_ds, return_dict=True, verbose=0)
    test_metrics = model.evaluate(test_ds, return_dict=True, verbose=0)
    print(f"VAL  -> loss={val_metrics['loss']:.4f}  acc={val_metrics['accuracy']:.4f}")
    print(f"TEST -> loss={test_metrics['loss']:.4f} acc={test_metrics['accuracy']:.4f}")
    return val_metrics, test_metrics

def main():
    train_ds, val_ds, test_ds = util.parse_dataset()

    model_a = ModelA(num_classes=100, dropout=0.3, l2=0.001)
    model_b = ModelB(num_classes=100, dropout=0.4, l2=0.0005)

    run_model(model_a, "ModelA", train_ds, val_ds, test_ds, learning_rate=1e-3)
    run_model(model_b, "ModelB", train_ds, val_ds, test_ds, learning_rate=3e-4)

if __name__ == '__main__':
    main()
