import tensorflow as tf
import tensorflow_datasets as tfds

def load_cifar100(batch_size=128):
    """Load CIFAR-100 and return train, val, test datasets with one-hot labels."""
    (train_ds, val_ds, test_ds), info = tfds.load(
        "cifar100",
        split=["train[:90%]", "train[90%:]", "test"],
        as_supervised=True,
        with_info=True
    )

    num_classes = info.features["label"].num_classes

    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.one_hot(label, num_classes)
        return image, label

    def augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_crop(tf.image.resize_with_crop_or_pad(image, 36, 36), size=[32, 32, 3])
        return image, label

    train_ds = (train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                         .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
                         .shuffle(10000)
                         .batch(batch_size)
                         .prefetch(tf.data.AUTOTUNE))

    val_ds = (val_ds.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE))
    test_ds = (test_ds.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE))

    return train_ds, val_ds, test_ds, num_classes
