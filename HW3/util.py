import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import math
import seaborn as sns

CLASS_NAMES = ["negative", "positive"]

# These will be set inside parse_dataset()
VOCAB_SIZE = None
SEQ_LEN = None


def parse_dataset(
    max_tokens: int = 10000,
    sequence_length: int = 500,
    batch_size: int = 64,
    buffer_size: int = 10000,
):
    DATA_DIR = "./tensorflow-datasets/"

    raw_train_ds = tfds.load(
        "imdb_reviews",
        split="train",
        data_dir=DATA_DIR,
        as_supervised=True,
    )

    train_ds_raw = tfds.load(
        "imdb_reviews",
        split="train[:80%]",
        data_dir=DATA_DIR,
        as_supervised=True,
    )
    val_ds_raw = tfds.load(
        "imdb_reviews",
        split="train[80%:]",
        data_dir=DATA_DIR,
        as_supervised=True,
    )
    test_ds_raw = tfds.load(
        "imdb_reviews",
        split="test",
        data_dir=DATA_DIR,
        as_supervised=True,
    )

    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=sequence_length,
    )

    text_only_ds = raw_train_ds.map(lambda text, label: text)
    vectorize_layer.adapt(text_only_ds)

    vocab = vectorize_layer.get_vocabulary()
    vocab_size = len(vocab)
    seq_len = sequence_length

    global VOCAB_SIZE, SEQ_LEN
    VOCAB_SIZE = vocab_size
    SEQ_LEN = seq_len

    def preprocess_text(text, label):
        text_vec = vectorize_layer(text)
        return text_vec, label

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = (
        train_ds_raw
        .map(preprocess_text, num_parallel_calls=AUTOTUNE)
        .shuffle(buffer_size)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    val_ds = (
        val_ds_raw
        .map(preprocess_text, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    test_ds = (
        test_ds_raw
        .map(preprocess_text, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    return train_ds, val_ds, test_ds, vocab_size, seq_len

# 2x2 confuction matrix
def confusion_matrix_plot(model, ds, filename, class_names=CLASS_NAMES):

    y_true, y_pred = [], []

    for x, y in ds:
        probs = model(x, training=False)
        probs = tf.convert_to_tensor(probs)

        if probs.shape[-1] == 1:
            # Binary sigmoid case
            p = tf.cast(probs > 0.5, tf.int32)
            p = tf.squeeze(p, axis=-1)
        else:
            # Multi-class softmax case
            p = tf.argmax(probs, axis=-1)

        y_true.extend(y.numpy())
        y_pred.extend(p.numpy())

    num_classes = len(class_names)
    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes).numpy()

    # Normalize
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (normalized)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    return cm


# 95% confidence interval
def confidence_interval(acc, n):
    z = 1.96  # z value for 95% confidence
    stderr = math.sqrt(acc * (1 - acc) / n)
    lower_bound = max(0.0, acc - z * stderr)
    upper_bound = min(1.0, acc + z * stderr)
    return lower_bound, upper_bound
