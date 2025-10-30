import tensorflow as tf
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
        if training:
            image = data_augmentation(image, training=True)
        label = tf.one_hot(label, depth=100)     
        return image, label

    BATCH_SIZE = 32
    BUFFER_SIZE = 1024

    # Prepare the datasets
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
