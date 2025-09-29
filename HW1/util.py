def parse_dataset():
    import numpy as np                 # to use numpy arrays
    import tensorflow as tf            # to specify and run computation graphs
    import tensorflow_datasets as tfds # to load training data
    import matplotlib.pyplot as plt    # to visualize data and draw plots
    from tqdm import tqdm              # to track progress of loops
    import os


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
    
    print(train_ds)
    print("dataset parsed")

    return train_ds, val_ds, test_ds