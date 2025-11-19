import tensorflow as tf

# Attention over time steps
class Attention(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.score_dense = tf.keras.layers.Dense(1)

    def call(self, inputs, mask=None):
        scores = self.score_dense(inputs)         
        scores = tf.squeeze(scores, axis=-1)       

        if mask is not None:
            scores += (1.0 - tf.cast(mask, tf.float32)) * -1e9

        weights = tf.nn.softmax(scores, axis=1)    
        weights_expanded = tf.expand_dims(weights, -1)
        context = tf.reduce_sum(inputs * weights_expanded, axis=1)
        return context


class Model(tf.keras.Model):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        rnn_units=128,
        dense_units=64,
        rnn_type="lstm",        # "lstm" or "gru"
        bidirectional=True,
        dropout_rate=0.5,
        l2_reg=1e-4,
    ):
        super().__init__()

        reg = tf.keras.regularizers.l2(l2_reg) if l2_reg and l2_reg > 0 else None

        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            embeddings_regularizer=reg,
            mask_zero=True,
            name="embedding",
        )

        if rnn_type.lower() == "gru":
            rnn_cls = tf.keras.layers.GRU
        else:
            rnn_cls = tf.keras.layers.LSTM

        rnn_layer = rnn_cls(
            rnn_units,
            return_sequences=True,
            kernel_regularizer=reg,
            recurrent_regularizer=reg,
            dropout=dropout_rate,
            name=f"{rnn_type}_layer",
        )

        if bidirectional:
            self.rnn = tf.keras.layers.Bidirectional(rnn_layer, name="bidirectional_rnn")
        else:
            self.rnn = rnn_layer

        self.attention = Attention()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense = tf.keras.layers.Dense(
            dense_units,
            activation="relu",
            kernel_regularizer=reg,
            name="dense",
        )
        self.output_layer = tf.keras.layers.Dense(
            1,
            activation="sigmoid",
            name="output",
        )

    def call(self, inputs, training=False):
        # inputs: [batch, seq_len] of token IDs
        x = self.embedding(inputs)              # [batch, time, emb]
        mask = self.embedding.compute_mask(inputs)
        x = self.rnn(x, training=training)      # [batch, time, rnn_units*(1 or 2)]
        context = self.attention(x, mask=mask)  # [batch, features]
        x = self.dropout(context, training=training)
        x = self.dense(x)
        x = self.dropout(x, training=training)
        return self.output_layer(x)
