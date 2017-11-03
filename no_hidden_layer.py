# To test the model type the following command
# python3 no_hidden_layer.py <iterations> <learning rate>

import os
import sys
import tensorflow as tf
import numpy as np
from subprocess import call

TRAIN = 'train/'
VALIDATE = 'valid/'
LABELS = 'labels.txt'
NUM_CLASSES = 104 #Different types of images in the dataset.

ITERS = 1000 #No. of iterations
LEARNING_RATE = 0.5

try:
    if len(sys.argv) > 1: #Two command line arguments are accepted as input i) iterations ii) learning rate
        ITERS = int(sys.argv[1])
        LEARNING_RATE = float(sys.argv[2])
except:
    pass #do nothing

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

print "All data loaded"

NUM_FEATURES = len(x_train[0]) 

#Number of different pixels for a given image
x = tf.placeholder(tf.float32, [None, NUM_FEATURES])

#Number of different kind of outputs = NUM_CLASSES 
y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

#weights for the 1st layer
W = tf.Variable(tf.zeros([NUM_FEATURES, NUM_CLASSES]))

#bias for the first layet o/p
b = tf.Variable(tf.zeros([NUM_CLASSES]))

#final o/p
y = tf.nn.softmax(tf.matmul(x, W) + b)

#calculating loss i.e variation of the model o/p from the actual o/p
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#This modifies each variable according to the magnitude of the derivation of loss with respect to that variable
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

#creating an operation for initializing the placeholders i.e x--> inputs  and y_--> outputs
init = tf.initialize_all_variables()

sess = tf.Session() #creates a session object
sess.run(init) 

print "Starting training for " + str(ITERS) + " iterations with learning rate " + str(LEARNING_RATE)

#traing the model for the given 1000 iteration and calculate the correct value of the weights and biases
for i in range(ITERS):
    sess.run(train_step, feed_dict={x: x_train, y_: y_train})

#calculating the accuracy of the model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: x_test, y_: y_test}))

os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (1, 1000))
