# Deep-neural-network

Used Tensorflow python library to build a deep neural network for Devanagari character recognition.

TensorFlow relies on a highly efficient C++ backend to do its computation. The connection to this backend is called a session. The common usage for TensorFlow programs is to first create a graph and then launch it in a session.

Goal was to design a feed-forward network for handwritten character regonition of Devanagari script.

To get an idea of working of tensorflow watch this video:

    https://www.youtube.com/watch?v=yX8KuPZCAMo&t=258s
    
Read more:
           
    https://www.tensorflow.org/versions/r0.7/tutorials/mnist/beginners/index.html        

The labeled dataset was provided for training the model(i.e Test data) were consists of 10,000 PNG format images(320x320). 
After the training process our model predicts the correct label for the image provided from the validation data set.

    What is a labeled data? 
    Labeled data is a group of samples that have been tagged with one or more labels. Labeling typically 
    takes a set of unlabeled data and augments each piece of that unlabeled data with meaningful tags 
    that are informative. After obtaining a labeled dataset, machine learning models can be applied to 
    the data so that new unlabeled data can be presented to the model and a likely label can be guessed 
    or predicted for that piece of unlabeled data.

### Download dataset from the following links:
     
   a) Download train data set from this link
     
        https://drive.google.com/file/d/0BzIqj5JgNb5RRlo1aUwyTDNEdzg/view 

   b) Download validation data set from this link
    
        https://drive.google.com/open?id=0BzIqj5JgNb5Ra0duNW95UDNHN0U 

The png image given in dataset is read using python package: scikit-image, it converts the png image into numpy-ndarray

### Following analysis was done:
- Effect of increasing the number of hidden units in our artificial neural network.
- Effect of varying(increasing) the learning rate.
- Effect of changing the actiavation function type from Ramp to Sigmoid to Hyperbolic tangent activation function.
- Effect of changing the optimizer(We will replace the steepest gradient descent optimizer with the more sophisticated 
  ADAM optimizer)
