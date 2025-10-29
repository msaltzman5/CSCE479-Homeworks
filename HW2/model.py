import tensorflow as tf

def cnn_model_1(input_shape=(32,32,3), num_classes=100, l2=1e-4, dropout=0.3):
    """Simple CNN with two conv+pool layers and one dense layer."""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2), input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model


def cnn_model_2(input_shape=(32,32,3), num_classes=100, l2=5e-4, dropout=0.5):
    """Deeper CNN with three conv+pool layers and one dense layer."""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2), input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(l2)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model
