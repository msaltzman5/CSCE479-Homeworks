# Requirements: 
#   - # of layers
#   - Different hyperparameters (need to have default values and overwrite if needed)
#   - Different regularizations

import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self, hidden_units=[256, 128], dropout_rates=[0.3, 0.5], num_classes=10):
        # Inherit from tf.keras.Model
        super(Model, self).__init__()

        assert len(hidden_units) == len(dropout_rates), \
            "hidden_units and dropout_rates lists must have the same length."

        self.hidden_layers = []
        for units, rate in zip(hidden_units, dropout_rates):
            self.hidden_layers.append(tf.keras.layers.Dense(units, activation="relu"))
            if rate > 0.0:
                self.hidden_layers.append(tf.keras.layers.Dropout(rate))

        # Output layer: size = number of classes, activation = softmax
        self.output_layer = tf.keras.layers.Dense(num_classes, activation="softmax")

    # From tf.keras.Model
    def call(self, inputs, training=False):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        return self.output_layer(x)