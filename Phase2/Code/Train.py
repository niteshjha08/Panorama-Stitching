#!/usr/bin/python3

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

from re import L
from unittest.mock import patch
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import sys
import os
import glob
# import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import HomographyModel
from Network.UnsupervisedNetwork import UnsupervisedModel
from Misc.MiscUtils import *
from Misc.DataUtils import *
import numpy as np
import time
import argparse
import shutil
# from StringIO import StringIO
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
from Misc.TFSpatialTransformer import *
from sklearn.utils import shuffle
from math import ceil
from keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from GenerateData import GeneratePatches


# Don't generate pyc codes
sys.dont_write_bytecode = True
config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

def generator(ImagesPath,n_images,batch_size,image_names):
    
    # Infinite loop which ends when epochs specified is completed
    while True:
        # Creating batches
        patches=[]
        labels = []
        count=0
        while(count<batch_size):
            retval, patch_A, patch_B, H_4pt, Ca,Cb,_=GeneratePatches(ImagesPath,image_names)
            if(retval):
                count+=1
                patchstack = np.dstack((patch_A,patch_B))
                patches.append(patchstack)
                
                X = H_4pt[:,0]
                Y = H_4pt[:,1]
           
                H4_flat = np.hstack((X,Y))
                labels.append(H4_flat)
                
        patches = np.array(patches)
        labels = np.array(labels)

        yield patches,labels

def unsupervised_batch_generator(ImagesPath,batch_size,image_names):
    patches=[]
    labels = []
    count=0
    img_1=[]
    Ca_batch = []
    patch_b_batch=[]
    while(count<batch_size):
        retval, patch_A, patch_B, H_4pt, Ca,Cb,img=GeneratePatches(ImagesPath,image_names)
        
        if(retval):
            count+=1
            patchstack = np.dstack((patch_A,patch_B))
            patches.append(patchstack)
            
            X = H_4pt[:,0]
            Y = H_4pt[:,1]
        
            H4_flat = np.hstack((X,Y))
            labels.append(H4_flat)
            img_1.append(img.reshape(img.shape[0],img.shape[1],1))
            Ca_batch.append(Ca)
            patch_b_batch.append(patch_B.reshape(128,128,1))
            
    patches = np.array(patches)
    labels = np.array(labels)
    img_1 = np.array(img_1)
    patch_b_batch = np.array(patch_b_batch)
    Ca_batch = np.array(Ca_batch)

    patches = patches/255.0
    patch_b_batch = patch_b_batch/255.0
    img_1 = img_1/255.0
    return patches,Ca_batch,patch_b_batch,img_1

def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print('Number of Epochs Training will run for ' + str(NumEpochs))
    print('Factor of reduction in training data is ' + str(DivTrain))
    print('Mini Batch Size ' + str(MiniBatchSize))
    print('Number of Training Images ' + str(NumTrainSamples))
    if LatestFile is not None:
        print('Loading latest checkpoint with the name ' + LatestFile)              


def L2_loss(y_true, y_pred):
    y_true=tf.cast(y_true, tf.float32)
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1, keepdims=True))


def TrainOperation(CheckPointPath,LogsPath):
    """
    Inputs: 
    ImgPH is the Input Image placeholder
    LabelPH is the one-hot encoded label placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainLabels - Labels corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
	ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """      
    # Creating folders for checkpoint and logs, if they don't exist
    if not (os.path.isdir(CheckPointPath)):
        os.makedirs(CheckPointPath)

    if not (os.path.isdir(LogsPath)):
        os.makedirs(LogsPath)

    # training image path
    ImagesPath=r"./../Data/Train"
    image_names=[]
    for dir,subdir,files in os.walk(ImagesPath):
        for file in files:
            image_names.append(file)

    ckpt_filename = CheckPointPath + "weights - {epoch:02d}.ckpt"
    ckpt = ModelCheckpoint(ckpt_filename,monitor='loss',mode=min,save_best_only=True,save_weights_only=True,verbose=1,save_freq=15650)

    model = HomographyModel()

    BATCH_SIZE = 64
    EPOCHS = 200
    N_IMAGES=20000
    # ImagesPath,n_images,batch_size,image_names
    train_generator = generator(ImagesPath,N_IMAGES,BATCH_SIZE,image_names)
    print(model.summary())
    
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001,momentum=0.9)

    model.compile(loss=L2_loss,optimizer=optimizer,metrics=['mean_absolute_error'])
    steps_per_epoch = ceil(N_IMAGES/BATCH_SIZE)
    progress_callback = model.fit_generator(train_generator,steps_per_epoch=steps_per_epoch,epochs=EPOCHS,verbose=1,callbacks=[ckpt])
    losses = progress_callback.history["loss"]
    errors = progress_callback.history['mean_absolute_error']

    np.savetxt(LogsPath + "loss_history.txt", np.array(losses), delimiter=',')
    np.savetxt(LogsPath + "errors_history.txt", np.array(errors), delimiter=',')

    model.save(CheckPointPath+ 'supervised_learning_sgd_lr0.0001,m0.9_20k_run2.h5')

def UnsupervisedTrainOperation(batch_size,total_epochs,n_samples,ImagesPath,image_names):

    Ca_batch_ = tf.placeholder(tf.float32, shape=(batch_size, 4,2))
    patches_batch_= tf.placeholder(tf.float32, shape=(batch_size, 128, 128 ,2))
    patch_b_batch_ = tf.placeholder(tf.float32, shape=(batch_size, 128, 128, 1))
    img_a_ = tf.placeholder(tf.float32, shape=(batch_size, 240, 320, 1))
    pred_patchB_,true_patchB_ = UnsupervisedModel(patches_batch_,batch_size,Ca_batch_,patch_b_batch_, img_a_)
    Saver = tf.train.Saver()
    with tf.name_scope('Loss'):
        loss = tf.reduce_mean(tf.abs(pred_patchB_ - true_patchB_))

    with tf.name_scope('Adam'):
        Optimizer = tf.train.AdamOptimizer(learning_rate=1e-6).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        steps_per_epoch = ceil(n_samples/batch_size)
        loss_history=[]
        for epoch in range(total_epochs):
            for batch in range(steps_per_epoch):
                patches_batch, Ca_batch,patch_b_batch,img_a = unsupervised_batch_generator(ImagesPath,batch_size,image_names)
                
                _,curr_loss = sess.run([Optimizer,loss], feed_dict={patches_batch_:patches_batch,Ca_batch_:Ca_batch,\
                                                                        patch_b_batch_:patch_b_batch,img_a_:img_a})
                loss_history.append(curr_loss)
    # Saver.save(sess,save_path="./Unsupervised_model_Savepath")

        errors_unsupervised = open("errors_unsup_1e-6.txt",'w')
        errors_unsupervised.write(str(loss_history))
        errors_unsupervised.close()
    print(loss_history)


def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='/media/nitin/Research/Homing/SpectralCompression/COCO', help='Base path of images, Default:/media/nitin/Research/Homing/SpectralCompression/COCO')
    Parser.add_argument('--CheckPointPath', default='../Checkpoints/', help='Path to save Checkpoints, Default: ../Checkpoints/')
    Parser.add_argument('--ModelType', default='Unsup', help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup')
    Parser.add_argument('--NumEpochs', type=int, default=50, help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=1, help='Size of the MiniBatch to use, Default:1')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Parser.add_argument('--LogsPath', default='Logs/', help='Path to save Logs for Tensorboard, Default=Logs/')

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType

    # Setup all needed parameters including file reading
    # DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses = SetupAll(BasePath, CheckPointPath)



    # Find Latest Checkpoint File
    if LoadCheckPoint==1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    # ----------------------------- Supervised training operation --------------------------------------
    if(ModelType=='Sup'):
        TrainOperation(CheckPointPath,LogsPath)
    
    # ---------------------------Unsupervised training operation-------------------------------------
    else:
        image_names=[]
        ImagesPath="./../Data/Train"
        for dir,subdir,files in os.walk(ImagesPath):
            for file in files:
                image_names.append(file)
        UnsupervisedTrainOperation(batch_size=64,total_epochs=100,n_samples=5000,ImagesPath=ImagesPath,image_names=image_names)
    # -----------------------------------------------------------------------------------------------    
    
if __name__ == '__main__':
    main()
 
