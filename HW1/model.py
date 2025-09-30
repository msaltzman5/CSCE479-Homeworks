# Requirements: 
#   - # of layers
#   - Different hyperparameters (need to have default values and overwrite if needed)
#   - Different regularizations

import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self, hidden_units, learning_rate, dropout, l2, num_classes=10):
        # Inherit from tf.keras.Model
        super(Model, self).__init__()

        self.hidden_layers = [tf.keras.layers.Rescaling(1.0/255.0)]

        reg = tf.keras.regularizers.l2(l2) if l2 > 0 else None

        for hu in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(hu, activation="relu", kernel_regularizer=reg))
            # Even droput for all layers
            if dropout and dropout > 0:
                self.hidden_layers.append(tf.keras.layers.Dropout(dropout))

        # Output layer: size = number of classes, activation = softmax
        self.output_layer = tf.keras.layers.Dense(num_classes, activation="softmax")

    # From tf.keras.Model
    def call(self, inputs, training=False):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        return self.output_layer(x)