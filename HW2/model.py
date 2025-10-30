import tensorflow as tf

# Base CNN class template (for reuse)
class CNNModel(tf.keras.Model):
    def __init__(self, num_classes=100, dropout=0.3, l2=0.001):
        super(CNNModel, self).__init__()

        reg = tf.keras.regularizers.l2(l2) if l2 > 0 else None

        self.conv_layers = tf.keras.Sequential([])

        self.dropout = dropout
        self.reg = reg

        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=reg)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.fc2 = tf.keras.layers.Dense(num_classes, activation="softmax")

    def call(self, inputs, training=False):
        x = self.conv_layers(inputs, training=training)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.drop1(x, training=training)
        return self.fc2(x)


# Model A: Smaller CNN (Simple architecture)
class ModelA(CNNModel):
    def __init__(self, num_classes=100, dropout=0.3, l2=0.001):
        super(ModelA, self).__init__(num_classes, dropout, l2)

        self.conv_layers = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                                   kernel_regularizer=self.reg),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                                   kernel_regularizer=self.reg),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2,2)),

            tf.keras.layers.Dropout(self.dropout)
        ])


# Model B: Deeper CNN (More filters + BatchNorm)
class ModelB(CNNModel):
    def __init__(self, num_classes=100, dropout=0.4, l2=0.0005):
        super(ModelB, self).__init__(num_classes, dropout, l2)

        self.conv_layers = tf.keras.Sequential([
            # Block 1
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                                   kernel_regularizer=self.reg),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                                   kernel_regularizer=self.reg),
            tf.keras.layers.MaxPooling2D((2, 2)),

            # Block 2
            tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu',
                                   kernel_regularizer=self.reg),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu',
                                   kernel_regularizer=self.reg),
            tf.keras.layers.MaxPooling2D((2, 2)),

            # Block 3
            tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu',
                                   kernel_regularizer=self.reg),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Dropout(self.dropout)
        ])
