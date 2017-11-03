# Deep-neural-network

Used Tensorflow python library to build a deep neural network for Devanagari character recognition.
Goal was to design a feed-forward network for handwritten character regonition of Devanagari script.

The dataset used for training the model(i.e Test data) were consists of 10,000 PNG format images(320x320). 
After the training process our model predicts the correct label for the image provided from the validation data set.

The png image given in dataset is read using python package: scikit-image, it converts the png image into numpy-ndarray

### Following analysis was done:
- Effect of increasing the number of hidden units in our artificial neural network.
- Effect of varying(increasing) the learning rate.
- Effect of changing the actiavation function type from Ramp to Sigmoid to Hyperbolic tangent activation function.
