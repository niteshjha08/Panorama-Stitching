#!/usr/bin/python3
from tensorflow.keras.utils import Sequence
import argparse
import sys
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
# from GenerateData import ImagesPath
from Misc.MiscUtils import *
from Misc.DataUtils import *
from tensorflow.keras.models import load_model
from GenerateData import GeneratePatches

# import tensorflow as tf
import tensorflow.compat.v1 as tf
import tensorflow as tf2
from keras import backend as K
tf.disable_v2_behavior()
def L2_loss(y_true, y_pred):
    # tf.cast(y_pred|y_true, tf.float32|tf.int64)
    # tf.cast(y_true, tf.int64)
    
    # y_pred=tf.cast(y_pred, tf.int64)
    y_true=tf.cast(y_true, tf.float32)

    print("type:",y_pred.dtype)
    print("type:",y_true.dtype)

    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1, keepdims=True))
# custom_objects={'loss_max':L2_loss}

if __name__=="__main__":
    # test_idx=7
    # model = load_model('./supervised_learning_adam_lr0.0001_L2_21_2_22.h5',custom_objects={'L2_loss':L2_loss})
    # patchApath='./../Data/PatchA/'+str(test_idx)+'.jpg'
    # patchA=cv2.imread(patchApath,0)
    # patchBpath='./../Data/PatchB/'+str(test_idx)+'.jpg'
    # patchB=cv2.imread(patchBpath,0)
    # print(patchA.shape)
    # print(patchB.shape)

    # input=np.dstack((patchA,patchB))
    # input = np.expand_dims(input,axis=0)
    # print(input.shape)

    # H4pt=model.predict(input)
    # print("predicted:",H4pt)
    # cv2.imshow('a',patchA)
    # cv2.imshow('b',patchB)
    # cv2.waitKey(0)
    # H4_list=np.load('./../Data/H4/H4.npy')
    # print("ground truth: ",H4_list[test_idx])

    # Visualization script of Ca, Cb, and Cb_pred

    model = load_model('./../Checkpoints2/supervised_learning_sgd_lr0.0001,m0.9_20k.h5',custom_objects={'L2_loss':L2_loss})
    ImagesPath = r"./../Data/Test/"
    error=0
    l2_loss=0
    img_names=os.listdir(ImagesPath)
    for i in range(1,len(img_names)):
        image_path = str(i)+'.jpg'
        image=cv2.imread(ImagesPath + image_path)
    # print(ImagesPath + image_path)
    # cv2.imshow('image',image)
    # cv2.waitKey(0)
        temp_list=[image_path]
        retval, patch_A, patch_B, H_4pt_actual, Ca,Cb,_=GeneratePatches(ImagesPath,temp_list)
        input=np.dstack((patch_A,patch_B))
        input = np.expand_dims(input,axis=0)
        H_4pt_pred=model.predict(input)[0]
        # print(H_4pt_pred)
        H_4pt_pred = np.vstack((H_4pt_pred[:4],H_4pt_pred[4:])).T
        # print(H_4pt_pred)
        Cb_pred = Ca + H_4pt_pred
        Cb_pred = (Cb_pred).astype(int)
        error+=np.sum(np.abs(Cb_pred-Cb))
        l2_loss += np.sqrt(np.sum(np.abs(Cb_pred-Cb)**2))
        # print("CB: \n",Cb)
        # print("CB_pred: \n",Cb_pred)
        # print("errror:\n",np.sum(np.abs(Cb_pred-Cb)))
        # cv2.polylines(image,[Cb],True,(255,0,0),2)
        # cv2.polylines(image,[Ca],True,(0,0,255),2)

        # cv2.polylines(image,[Cb_pred],True,(0,255,0),2)
        # print(image.shape)
        # cv2.imshow('image',image)
        # cv2.waitKey(0)
        # cv2.waitKey(0)

        # cv2.waitKey(0)

        # -------------------------------------------------------------


    error = error/len(img_names)
    print("average error is:",error/4)
    print("average L2 loss is:",l2_loss/len(img_names))








    
        