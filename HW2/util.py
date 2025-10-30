import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import tensorflow_datasets as tfds

def parse_dataset():
    DATA_DIR = './tensorflow-datasets/'

    ds = tfds.load('cifar100', data_dir=DATA_DIR, shuffle_files=True, as_supervised=True)

    train_ds = tfds.load('cifar100', split='train[:90%]', data_dir=DATA_DIR, as_supervised=True)
    val_ds   = tfds.load('cifar100', split='train[-10%:]', data_dir=DATA_DIR, as_supervised=True)
    test_ds  = tfds.load('cifar100', split='test', data_dir=DATA_DIR, as_supervised=True)
    
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
    ])

    # Preprocess images by normalizing RGB values of images and one-hot encoding labels
    # https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
    MEAN = [0.5071, 0.4867, 0.4408]
    STD = [0.2675, 0.2565, 0.2761]
    def preprocess_img(image, label, training=False):
        image = tf.cast(image, tf.float32) / 255.0
        image = (image - MEAN) / STD
        # if training:
        #     image = data_augmentation(image, training=True)
        label = tf.one_hot(label, depth=100)     
        return image, label

    BATCH_SIZE = 32
    BUFFER_SIZE = 1024

    train_ds = (train_ds
                .map(lambda x, y: preprocess_img(x, y, training=True), num_parallel_calls=tf.data.AUTOTUNE)
                .shuffle(BUFFER_SIZE)
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE))

    val_ds = (val_ds
              .map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(BATCH_SIZE)
              .prefetch(tf.data.AUTOTUNE))

    test_ds = (test_ds
               .map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
               .batch(BATCH_SIZE)
               .prefetch(tf.data.AUTOTUNE))

    return train_ds, val_ds, test_ds

def confusion_matrix_plot(model, ds, filename, class_names=None, max_labels=5):
    y_true, y_pred = [], []

    for x, y in ds:
        preds = model(x, training=False)
        y_pred_batch = tf.argmax(preds, axis=-1).numpy()

        # Handle one-hot labels
        if y.shape[-1] > 1:
            y_true_batch = tf.argmax(y, axis=-1).numpy()
        else:
            y_true_batch = y.numpy()

        y_true.extend(y_true_batch)
        y_pred.extend(y_pred_batch)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    num_classes = np.max(y_true) + 1

    # Load class names from TFDS
    if class_names is None:
        info = tfds.builder("cifar100").info
        class_names = info.features["label"].names

    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes).numpy()

    # Normalize
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)  # divide-by-zero case

    # Show only top N
    if num_classes > max_labels:
        totals = cm.sum(axis=1)
        top_idx = np.argsort(totals)[-max_labels:]
        cm_norm = cm_norm[top_idx][:, top_idx]
        class_names = [class_names[i] for i in top_idx]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=False, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Normalized Confusion Matrix (top 5 {num_classes} classes)")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    return cm

def confidence_interval(acc, n):
    z = 1.96  # 95% confidence
    stderr = math.sqrt(acc * (1 - acc) / n)
    lower = max(0.0, acc - z * stderr)
    upper = min(1.0, acc + z * stderr)
    return lower, upper