import numpy as np                 
import tensorflow as tf            
import tensorflow_datasets as tfds 
import matplotlib.pyplot as plt    
from tqdm import tqdm              
import os
import math
import seaborn as sns

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

def parse_dataset():
    DATA_DIR = './tensorflow-datasets/'

    ds = tfds.load('fashion_mnist', data_dir=DATA_DIR, shuffle_files=True, as_supervised=True) # this loads a dict with the datasets

    train_ds = tfds.load('fashion_mnist', split='train[:90%]', data_dir=DATA_DIR, as_supervised=True)
    val_ds = tfds.load('fashion_mnist', split='train[-10%:]', data_dir=DATA_DIR, as_supervised=True)
    test_ds = tfds.load('fashion_mnist', split='test', data_dir=DATA_DIR, as_supervised=True)

    # Flatten image 
    def preprocess_img(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.reshape(image, [-1])
        return image, label

    BATCH_SIZE = 32
    BUFFER_SIZE = 1024

    train_ds = (train_ds
                .map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
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

def confusion_matrix_plot(model, ds, filename, class_names=CLASS_NAMES):
    y_true, y_pred = [], []
    for x, y in ds:
        p = tf.argmax(model(x, training=False), axis=-1)
        y_true.extend(y.numpy())
        y_pred.extend(p.numpy())
    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=len(class_names)).numpy()
    
    # Normalize so values are percentages
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Plot
    plt.figure(figsize=(8,6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix (normalized)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    return cm

def confidence_interval(acc, n):
    z = 1.96  # z value for 95% confidence
    stderr = math.sqrt(acc * (1 - acc) / n)  # standard error formula
    lower_bound = max(0.0, acc - z * stderr)
    upper_bound = min(1.0, acc + z * stderr)
    return lower_bound, upper_bound


