import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = './tensorflow-datasets/'

# Load MNIST with train/validation/test splits
train_ds = tfds.load('mnist', split='train[:90%]', data_dir=DATA_DIR, as_supervised=True)
val_ds = tfds.load('mnist', split='train[-10%:]', data_dir=DATA_DIR, as_supervised=True)
test_ds = tfds.load('mnist', split='test', data_dir=DATA_DIR, as_supervised=True)

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [-1])  # flatten 28x28 -> 784
    return image, label

BATCH_SIZE = 64
train_ds = train_ds.map(preprocess).shuffle(1024).batch(BATCH_SIZE)
val_ds = val_ds.map(preprocess).batch(BATCH_SIZE)
test_ds = test_ds.map(preprocess).batch(BATCH_SIZE)

# model1 (1 hidden layer)
model1 = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10)
])
optimizer1 = tf.keras.optimizers.Adam()

# Training loop using explicit sparse_softmax_cross_entropy_with_logits
loss_history1 = []
accuracy_history1 = []

EPOCHS = 5
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    for batch in train_ds:
        x, labels = batch
        with tf.GradientTape() as tape:
            logits = model1(x)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            scalar_loss = tf.reduce_mean(loss)
        grads = tape.gradient(scalar_loss, model1.trainable_variables)
        optimizer.apply_gradients(zip(grads, model1.trainable_variables))
        loss_history1.append(scalar_loss.numpy())
        predictions = tf.argmax(logits, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
        accuracy_history1.append(accuracy.numpy())

# Evaluate on test data
test_accuracy = tf.keras.metrics.Accuracy()
for x, labels in test_ds:
    logits = model1(x)
    preds = tf.argmax(logits, axis=1)
    test_accuracy.update_state(labels, preds)
print("Model 1 Test Accuracy:", test_accuracy.result().numpy())

# model2 (2 hidden layers with different activations)
model2 = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(10)
])
optimizer2 = tf.keras.optimizers.Adam()

loss_history2 = []
accuracy_history2 = []

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS} (Model 2)")
    for batch in train_ds:
        x, labels = batch
        with tf.GradientTape() as tape:
            logits = model2(x)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            scalar_loss = tf.reduce_mean(loss)
        grads = tape.gradient(scalar_loss, model2.trainable_variables)
        optimizer2.apply_gradients(zip(grads, model2.trainable_variables))
        loss_history2.append(scalar_loss.numpy())
        predictions = tf.argmax(logits, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
        accuracy_history2.append(accuracy.numpy())

# Evaluate model2 on test data
test_accuracy2 = tf.keras.metrics.Accuracy()
for x, labels in test_ds:
    logits = model2(x)
    preds = tf.argmax(logits, axis=1)
    test_accuracy2.update_state(labels, preds)
print("Model 2 Test Accuracy:", test_accuracy2.result().numpy())

# model3 (3 smaller hidden layers with same activation)
model3 = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(25, activation='relu'),
    tf.keras.layers.Dense(10)
])
optimizer3 = tf.keras.optimizers.Adam()

loss_history3 = []
accuracy_history3 = []

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS} (Model 3)")
    for batch in train_ds:
        x, labels = batch
        with tf.GradientTape() as tape:
            logits = model3(x)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            scalar_loss = tf.reduce_mean(loss)
        grads = tape.gradient(scalar_loss, model3.trainable_variables)
        optimizer3.apply_gradients(zip(grads, model3.trainable_variables))
        loss_history3.append(scalar_loss.numpy())
        predictions = tf.argmax(logits, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32))
        accuracy_history3.append(accuracy.numpy())

# Evaluate model3 on test data
test_accuracy3 = tf.keras.metrics.Accuracy()
for x, labels in test_ds:
    logits = model3(x)
    preds = tf.argmax(logits, axis=1)
    test_accuracy3.update_state(labels, preds)
print("Model 3 Test Accuracy:", test_accuracy3.result().numpy())

# Plot training loss for both models
plt.plot(history1.history['loss'], label='Model1 Train Loss')
plt.plot(history1.history['val_loss'], label='Model1 Val Loss')
plt.plot(history2.history['loss'], label='Model2 Train Loss')
plt.plot(history2.history['val_loss'], label='Model2 Val Loss')
plt.plot(history3.history['loss'], label='Model3 Train Loss')
plt.plot(history3.history['val_loss'], label='Model3 Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("plot.png")
