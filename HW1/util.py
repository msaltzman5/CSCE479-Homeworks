import numpy as np                 
import tensorflow as tf            
import tensorflow_datasets as tfds 
import matplotlib.pyplot as plt    
from tqdm import tqdm              
import os

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

def confusion_matrix(model, ds):
    y_true, y_pred = [], []
    for x, y in ds:
        p = tf.argmax(model(x, training=False), axis=-1)
        y_true.append(y)
        y_pred.append(p)
    y_true = tf.concat(y_true, axis=0)
    y_pred = tf.concat(y_pred, axis=0)
    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=10)
    return cm