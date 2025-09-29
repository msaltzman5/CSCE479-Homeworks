# We'll start with our library imports...
from __future__ import print_function

import numpy as np  # to use numpy arrays
import tensorflow as tf  # to specify and run computation graphs
import matplotlib.pyplot as plt  # for plotting

learning_rate = 0.5
num_iterations = 20

# the optimizer allows us to apply gradients to update variables
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Create a fixed matrix, A
A = tf.random.normal([8, 8])
# Create x using an arbitrary initial value
x = tf.Variable(tf.ones([8, 1]))
# Create a fixed vector b
b = tf.random.normal([8, 1])

# Lists to track metrics
squared_errors = []
grad_norms = []

# Check the initial values
print("A:", A.numpy())
print("b:", b.numpy())

print("Initial x:", x.numpy())
print("Ax:", (A @ x).numpy())
print()

# We want Ax - b = 0, so we'll try to minimize its value
for step in range(num_iterations):
    print("Iteration", step)
    with tf.GradientTape() as tape:
        # Calculate A*x
        product = tf.matmul(A, x)
        # calculat the loss value we want to minimize
        # what happens if we don't use the square here?
        difference_sq = tf.math.square(product - b)
        squared_error = tf.norm(tf.math.sqrt(difference_sq)).numpy()
        print("Squared error:", squared_error)
        # calculate the gradient
        grad = tape.gradient(difference_sq, [x])
        grad_norm = tf.norm(grad[0]).numpy()
        print("Gradients:")
        print(grad)
        # update x
        optimizer.apply_gradients(zip(grad, [x]))
        print()

    # Save for plotting
    squared_errors.append(squared_error)
    grad_norms.append(grad_norm)

# Check the final values
print("Optimized x", x.numpy())
print("Ax", (A @ x).numpy())  # Should be close to the value of b

# Plot metrics
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(squared_errors, marker='o')
plt.title("Squared Error over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Squared Error")

plt.subplot(1, 2, 2)
plt.plot(grad_norms, marker='o', color='orange')
plt.title("Gradient Norm over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Gradient Norm")

plt.tight_layout()
plt.savefig("plot.png")
