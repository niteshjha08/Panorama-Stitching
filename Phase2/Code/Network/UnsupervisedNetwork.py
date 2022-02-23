"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import tensorflow as tf
import sys
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, InputLayer, BatchNormalization, Layer
# Don't generate pyc codes
sys.dont_write_bytecode = True

class TensorDLT(Layer):
    def __init__(self, corners, estimatedHomo):
        self.corners = corners
        self.estimatedHomo = estimatedHomo

    def call(self):
        h = 0
        for c in self.corners:
            h += np.dot(self.estimatedHomo, c)
        return h/len(self.corners)

# def HomographyModel(Img, ImageSize, MiniBatchSize):
def HomographyModel():

    """
    Inputs: 
    Img is a MiniBatch of the current image
    ImageSize - Size of the Image
    Outputs:
    prLogits - logits output of the network
    prSoftMax - softmax output of the network
    """

    #############################
    # Fill your network here!
    #############################
    input_shape = (128,128,2)

    model = Sequential()
    model.add(InputLayer(input_shape))

    # relu layer after batch normalization
    # two conv2d layers and maxpooling2d layer on 128x128 size
    model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2,2)))

    # two conv2d layers and maxpooling2d layer on 64x64
    model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2,2)))

    # two conv2d layers and maxpooling layer on 32x32
    model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2,2)))

    # two conv2d layers on 16x16
    model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(BatchNormalization())

    # Fully connected layer
    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(8))

    return model

    # return H4Pt

