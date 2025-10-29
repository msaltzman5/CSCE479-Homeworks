import tensorflow as tf
import tensorflow_datasets as tfds

def parse_dataset():
    DATA_DIR = './tensorflow-datasets/'

    # Load the CIFAR-100 dataset (includes train/test splits)
    ds = tfds.load('cifar100', data_dir=DATA_DIR, shuffle_files=True, as_supervised=True)

    # Partition train set into 90% train / 10% validation
    train_ds = tfds.load('cifar100', split='train[:90%]', data_dir=DATA_DIR, as_supervised=True)
    val_ds   = tfds.load('cifar100', split='train[-10%:]', data_dir=DATA_DIR, as_supervised=True)
    test_ds  = tfds.load('cifar100', split='test', data_dir=DATA_DIR, as_supervised=True)

    # Preprocess function for images and labels
    def preprocess_img(image, label):
        image = tf.cast(image, tf.float32) / 255.0        # normalize to [0,1]
        label = tf.one_hot(label, depth=100)              # convert to one-hot vector
        return image, label

    BATCH_SIZE = 32
    BUFFER_SIZE = 1024

    # Prepare the datasets
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
