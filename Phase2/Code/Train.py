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


# def generator(batch_size):
#     patchA_path = r"./../Data/PatchA"
#     patchB_path = r"./../Data/PatchB"
#     H4_path = r"./../Data/H4"
#     patchA_imgs = []
#     n_patches = 5000 # number of image patches(A and B) available
#     H4_list = np.load(H4_path + "/H4.npy",allow_pickle=True)
#     # print("list length:",len(H4_list))
#     # Infinite loop which ends when epochs specified is completed
#     while True:
#         # shuffle(filepaths)
#         # Creating batches
#         for i in range(0, n_patches-batch_size-1, batch_size):
#             patches=[]
#             labels = []
#             for batch_sample in range(batch_size):
#                 patchA_name = patchA_path + os.sep + str(i + batch_sample) + ".jpg"
#                 patchA = cv2.imread(patchA_name,0)
#                 # patchA=
#                 patchB_name = patchB_path + os.sep + str(i + batch_sample) + ".jpg"
#                 patchB = cv2.imread(patchB_name,0)

#                 patchstack = np.dstack((patchA,patchB))
#                 patches.append(patchstack)
             
#                 X = H4_list[i+batch_sample][:,0]
#                 Y = H4_list[i+batch_sample][:,1]
#                 H4_flat = np.hstack((X,Y))
#                 labels.append(H4_flat)
                
#                 # print("H4 flat:",H4_flat)
#                 # print(H4_list[i+batch_sample])
#                 # exit()
#             patches = np.array(patches)
#             labels = np.array(labels)

#             yield patches,labels

def generator(ImagesPath,n_images,batch_size,image_names):
    
    # Infinite loop which ends when epochs specified is completed
    while True:
        # shuffle(filepaths)

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
            img_1.append(img)
            Ca_batch.append(Ca)
            patch_b_batch.append(patch_B)
            
    patches = np.array(patches)
    labels = np.array(labels)
    img_1 = np.array(img_1)
    patch_b_batch = np.array(patch_b_batch)
    Ca_batch = np.array(Ca_batch)

    return patches,Ca_batch,patch_b_batch,img_1


def GenerateBatch(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize):
    """
    Inputs: 
    BasePath - Path to COCO folder without "/" at the end
    DirNamesTrain - Variable with Subfolder paths to train files
    NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
    TrainLabels - Labels corresponding to Train
    NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
    ImageSize - Size of the Image
    MiniBatchSize is the size of the MiniBatch
    Outputs:
    I1Batch - Batch of images
    LabelBatch - Batch of one-hot encoded labels 
    """
    I1Batch = []
    LabelBatch = []
    
    ImageNum = 0
    while ImageNum < MiniBatchSize:
        # Generate random image
        RandIdx = random.randint(0, len(DirNamesTrain)-1)
        
        RandImageName = BasePath + os.sep + DirNamesTrain[RandIdx] + '.jpg'   
        ImageNum += 1
    	
    	##########################################################
    	# Add any standardization or data augmentation here!
    	##########################################################

        I1 = np.float32(cv2.imread(RandImageName))
        Label = convertToOneHot(TrainLabels[RandIdx], 10)

        # Append All Images and Mask
        I1Batch.append(I1)
        LabelBatch.append(Label)
        
    return I1Batch, LabelBatch


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



# def TrainOperation(ImgPH, LabelPH, DirNamesTrain, TrainLabels, NumTrainSamples, ImageSize,
#                    NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
# 
#                    DivTrain, LatestFile, BasePath, LogsPath, ModelType):

def L2_loss(y_true, y_pred):
    # tf.cast(y_pred|y_true, tf.float32|tf.int64)
    # tf.cast(y_true, tf.int64)
    
    # y_pred=tf.cast(y_pred, tf.int64)
    y_true=tf.cast(y_true, tf.float32)

    # print("type:",y_pred.dtype)
    # print("type:",y_true.dtype)

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

    # Predict output with forward pass
    # prLogits, prSoftMax = HomographyModel(ImgPH, ImageSize, MiniBatchSize)
    model = HomographyModel()

    BATCH_SIZE = 64
    EPOCHS = 200
    N_IMAGES=20000
    # ImagesPath,n_images,batch_size,image_names
    train_generator = generator(ImagesPath,N_IMAGES,BATCH_SIZE,image_names)
    print(model.summary())
    # loss = tf.keras.losses.MeanSquaredError()
    
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001,momentum=0.9)
    #learning rate 0.001
    # SGD, momentum=0.09
    # nesterov = True
    # -------------------------------------
    # ckpt
    model.compile(loss=L2_loss,optimizer=optimizer,metrics=['mean_absolute_error'])
    steps_per_epoch = ceil(N_IMAGES/BATCH_SIZE)
    progress_callback = model.fit_generator(train_generator,steps_per_epoch=steps_per_epoch,epochs=EPOCHS,verbose=1,callbacks=[ckpt])
    losses = progress_callback.history["loss"]
    errors = progress_callback.history['mean_absolute_error']

    np.savetxt(LogsPath + "loss_history.txt", np.array(losses), delimiter=',')
    np.savetxt(LogsPath + "errors_history.txt", np.array(errors), delimiter=',')

    # model.save('supervised_learning_adam_lr0.0001_L2_21_2_22.h5')
    model.save(CheckPointPath+ 'supervised_learning_sgd_lr0.0001,m0.9_20k_run2.h5')






    # with tf.name_scope('Loss'):
    #     ###############################################
    #     # Fill your loss function of choice here!
    #     ###############################################
    #     loss = tf.reduce_mean()
    # with tf.name_scope('Adam'):
    # 	###############################################
    # 	# Fill your optimizer of choice here!
    # 	###############################################
    #     Optimizer = ...

    # # Tensorboard
    # # Create a summary to monitor loss tensor
    # tf.summary.scalar('LossEveryIter', loss)
    # # tf.summary.image('Anything you want', AnyImg)
    # # Merge all summaries into a single operation
    # MergedSummaryOP = tf.summary.merge_all()

    # # Setup Saver
    # Saver = tf.train.Saver()
    
    # with tf.Session() as sess:       
    #     if LatestFile is not None:
    #         Saver.restore(sess, CheckPointPath + LatestFile + '.ckpt')
    #         # Extract only numbers from the name
    #         StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
    #         print('Loaded latest checkpoint with the name ' + LatestFile + '....')
    #     else:
    #         sess.run(tf.global_variables_initializer())
    #         StartEpoch = 0
    #         print('New model initialized....')

    #     # Tensorboard
    #     Writer = tf.summary.FileWriter(LogsPath, graph=tf.get_default_graph())
            
    #     for Epochs in tqdm(range(StartEpoch, NumEpochs)):
    #         NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize/DivTrain)
    #         for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
    #             I1Batch, LabelBatch = GenerateBatch(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize)
    #             FeedDict = {ImgPH: I1Batch, LabelPH: LabelBatch}
    #             _, LossThisBatch, Summary = sess.run([Optimizer, loss, MergedSummaryOP], feed_dict=FeedDict)
                
    #             # Save checkpoint every some SaveCheckPoint's iterations
    #             if PerEpochCounter % SaveCheckPoint == 0:
    #                 # Save the Model learnt in this epoch
    #                 SaveName =  CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + 'model.ckpt'
    #                 Saver.save(sess,  save_path=SaveName)
    #                 print('\n' + SaveName + ' Model Saved...')

    #             # Tensorboard
    #             Writer.add_summary(Summary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
    #             # If you don't flush the tensorboard doesn't update until a lot of iterations!
    #             Writer.flush()

    #         # Save model every epoch
    #         SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
    #         Saver.save(sess, save_path=SaveName)
    #         print('\n' + SaveName + ' Model Saved...')
            

def UnsupervisedTrainOperation(batch_size,total_epochs,n_samples,ImagesPath,image_names):

    Ca_batch = tf.placeholder(tf.float32, shape=(batch_size, 4,2))
    patches_batch= tf.placeholder(tf.float32, shape=(batch_size, 128, 128 ,2))
    patch_b_batch = tf.placeholder(tf.float32, shape=(batch_size, 128, 128, 1))
    img_a = tf.placeholder(tf.float32, shape=(batch_size, 240, 320, 1))
    pred_patchB,true_patchB = UnsupervisedModel(patches_batch,batch_size,Ca_batch,patch_b_batch, img_a)
    Saver = tf.train.Saver()
    with tf.name_scope('Loss'):
        loss = tf.reduce_mean(tf.abs(pred_patchB - true_patchB))

    with tf.name_scope('Adam'):
        Optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        steps_per_epoch = ceil(n_samples/batch_size)
        for epoch in range(total_epochs):
            for batch in range(steps_per_epoch):
                patches_batch, Ca_batch,patch_b_batch,img_a = unsupervised_batch_generator(ImagesPath,batch_size,image_names)
                
                sess.run([Optimizer, loss], feed_dict={patches_batch:patches_batch,Ca_batch:Ca_batch,\
                                                                        patch_b_batch:patch_b_batch,img_a:img_a})


    Saver.save(sess,save_path="./Unsupervised_model_Savepath")


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

    # Pretty print stats
    # PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    # Define PlaceHolder variables for Input and Predicted output
    # ImgPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, ImageSize[0], ImageSize[1], ImageSize[2]))
    # LabelPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, NumClasses)) # OneHOT labels

    # TrainOperation(ImgPH, LabelPH, DirNamesTrain, TrainLabels, NumTrainSamples, ImageSize,
    #                 NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
    #                 DivTrain, LatestFile, BasePath, LogsPath, ModelType)

    # ----------------------------- Supervised training operation --------------------------------------
    # TrainOperation(CheckPointPath,LogsPath)
    # --------------------------------------------------------------------------------------------------
    # 
    # ---------------------------Unsupervised training operation-------------------------------------
    image_names=[]
    ImagesPath="./../Data/Train"
    for dir,subdir,files in os.walk(ImagesPath):
        for file in files:
            image_names.append(file)
    UnsupervisedTrainOperation(batch_size=32,total_epochs=3,n_samples=100,ImagesPath=ImagesPath,image_names=image_names)
    # -----------------------------------------------------------------------------------------------    
    
if __name__ == '__main__':
    main()
    # generator(32) 
