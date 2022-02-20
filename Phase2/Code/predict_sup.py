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
from Misc.MiscUtils import *
from Misc.DataUtils import *
from tensorflow.keras.models import load_model
if __name__=="__main__":
    model = load_model('supervised_learning.h5')
    patchApath='./../Data/PatchA/135.jpg'
    patchA=cv2.imread(patchApath,0)
    patchBpath='./../Data/PatchB/135.jpg'
    patchB=cv2.imread(patchBpath,0)
    print(patchA.shape)
    print(patchB.shape)

    input=np.dstack((patchA,patchB))
    print(input.shape)
    H4pt=model.predict(input)
    print(H4pt)
    cv2.imshow('a',patchA)
    cv2.imshow('b',patchB)
    cv2.waitKey(0)




    
        