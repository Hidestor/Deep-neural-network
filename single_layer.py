# To test the model type the following command
# python3 single_layer.py <iterations> <learning rate>

from __future__ import print_function

import os
import sys
import tensorflow as tf
import numpy as np
from subprocess import call


TRAINING_DATA = 'train_preprocessed.npy'
TRAINING_LABELS = 'train_preprocessed_labels.npy'
VALIDATION_DATA = 'valid_preprocessed.npy'
VALIDATION_LABELS = 'valid_preprocessed_labels.npy'

if not os.path.isfile(TRAINING_DATA):
    call(["python", "preprocess_data.py", "TRAIN"])

if not os.path.isfile(VALIDATION_DATA):
    call(["python", "preprocess_data.py", "VALID"])

if not os.path.isfile(TRAINING_LABELS):
    call(["python", "preprocess_labels.py", "TRAIN"])

if not os.path.isfile(VALIDATION_LABELS):
    call(["python", "preprocess_labels.py", "VALID"])

x_train = np.load('train_preprocessed.npy')
y_train = np.load('train_preprocessed_labels.npy')
x_test = np.load('valid_preprocessed.npy')
y_test = np.load('valid_preprocessed_labels.npy')


# Parameters
learning_rate = 0.005
batch_size = 1000  
n_hidden_1 = 1000  # 1st layer number of features

# Number of iterations
training_epochs = 100

display_step = 1
stddev = 0.01

# Network Parameters
n_input = len(x_train[0])
n_classes = 104

# tf Graph input
x = tf.placeholder("float", [None, n_input]) #input to the network
y = tf.placeholder("float", [None, n_classes]) #actual o/p of the network 


# Create model
def multilayer_perceptron(x, weights, biases):
    
    # Hidden layer with RELU activation / Ramp activation function 
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    
    #applying activation function to the o/p of layer1
    layer_1 = tf.nn.relu(layer_1)
    
    # Output layer with linear activation
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=stddev)),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes], stddev=stddev))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1], stddev=stddev)),
    'out': tf.Variable(tf.random_normal([n_classes], stddev=stddev))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer here using Adam optimization
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) 

# Initializing the variables
init = tf.initialize_all_variables()

print("Starting Optimization")

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(x_train) / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x = x_train[i * batch_size: (i + 1) * batch_size]
            batch_y = y_train[i * batch_size: (i + 1) * batch_size]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=",
                  "{:.9f}".format(avg_cost))
    print("Optimization Finished")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: x_test, y: y_test}))

call(['speech-dispatcher'])        #start speech dispatcher
call(['spd-say', '"your process has finished"'])
