#This module is used for converting the dataset into the form which could be directly used in a program
#in this we are storing the name of the image data cotained in the train dataset or validate dataset into a list and performing
#the required transformation into the image to extract the image data efficiently and correctly .

import sys
import os
import scipy
import numpy as np
from skimage import io # scikit-image is a collection of algorithms for image processing.

#Gaussian blur can be used in order to obtain a smooth grayscale digital image.
from scipy.ndimage.filters import gaussian_filter
from scipy.misc import toimage


BLUR_AMOUNT = 5
FINAL_SIZE = 80

TRAIN = 'train/' #train directory contains image data used for training the algorithm
VALIDATE = 'valid/' #valid directory contains image data used for checking validity of the algorithm 
PNG = '.png'    
LABELS = 'labels.txt'

DATASET = VALIDATE #for storing the chosen directory name, initially set to 'valid' direcory

try:
    if sys.argv[1] == 'TRAIN':
        print "Preprocessing training data"
        DATASET = TRAIN
    elif sys.argv[1] == 'VALID':
        print "Preprocessing validation data"
    else:
        print "Invalid argument .. quitting"
        sys.exit()
except:
    DATASET = VALIDATE

#getting the names of all images in the DATASET    
images = os.listdir(DATASET)

#deleting the labels.txt file from the images list
images.remove(LABELS) 

#storing all the file names without extension in a list
images = [int(image[:-4]) for image in images]  # -4 because last 4 characters will be .png

images.sort()

#converting file name from int to string
images = [str(image) for image in images] 


#This module takes the input of image in form of numpy ndarray and applies gaussian filter to it and finally stores 
#the pixel information in scaled down form
def process(image):
    
    # apply gaussian filter to image to make text wider
    image = gaussian_filter(image, sigma=BLUR_AMOUNT)
    
    # invert black and white because most of the image is white
    image = 255 - image
    
    # resize image to make it smaller
    image = scipy.misc.imresize(arr=image, size=(FINAL_SIZE, FINAL_SIZE))
    
    # scale down the value of each pixel
    image = image / 255.0
    
    # flatten the image array to a list
    return [item for sublist in image for item in sublist]


preprocessed = []


for item in images:
    #This code is there to read png image into numpy ndarray (matrix)
    image = np.array(io.imread(DATASET + item + PNG)) 
    
    image = process(image)
    preprocessed.append(image)

np.save(DATASET[:-1] + '_preprocessed', preprocessed)
