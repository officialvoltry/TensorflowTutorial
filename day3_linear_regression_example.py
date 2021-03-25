import numpy as np
# import tkinter
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
tf.disable_v2_behavior()

x_data = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)
y_label = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)

# plt.plot(x_data, y_label, '*')
print(np.random.rand(2))
m = tf.Variable(0.34)
b = tf.Variable(0.88)

error = 0

for x,y in zip(x_data, y_label):
    y_hat = m*x + b
    # Our predicted value
    error += (y - y_hat)**2
    # The cost we want to minimize 
    # We'll need to use optimizaion function the minimization

# Apply the Optimization Function
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

# Initialize variables
init = tf.global_variables_initializer()

# Create the session and run the computation

with tf.Session() as sess:
    sess.run(init)

    epochs = 100

    for i in range(epochs):
        
        sess.run(train)

    # Fetch back results
    final_scope, final_intercept = sess.run([m, b])

print(final_scope)
print(final_intercept)
