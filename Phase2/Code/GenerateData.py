#!/usr/bin/python3
import cv2
import numpy as np
import random
import math
import os


def GeneratePatches(ImagesPath,image_names, patch_size=128,rho=32):

    idx=random.randint(0,len(image_names)-1)
    print(idx)
    path=ImagesPath + os.sep + image_names[idx]
    img=cv2.imread(path)  
    
    M,N,_=img.shape

    if((M<patch_size+2*rho+1) | (N<patch_size+2*rho+1)):
        return False,_,_,_,_,_

    # Top left corner of patch selected
    # print("M rand is between range: ",rho," to ",M-patch_size-rho) 
    # print("N rand is between range: ",rho," to ",N-patch_size-rho)
    if(M<229 or N <229):
        return False,_,_,_,_,_

    Ca1=np.array([random.randint(rho,M-patch_size-rho),random.randint(rho,N-patch_size-rho)])

    Ca=np.array([[Ca1[1],Ca1[0]],
                 [Ca1[1]+patch_size,Ca1[0]],
                 [Ca1[1]+patch_size,Ca1[0]+patch_size ],
                 [Ca1[1],Ca1[0]+patch_size]])

    # Finding patch after random perturbations
    Cb=np.zeros(Ca.shape,dtype=np.int)

    for i in range(4):
        rx=random.randint(-rho,rho)
        ry=random.randint(-rho,rho)
        Cb[i,0]=Ca[i,0] + ry
        Cb[i,1]=Ca[i,1] + rx

    # Visualize patches and perspective transform:
    # cv2.polylines(img,[Ca],True,(0,0,255),5)
    # cv2.polylines(img,[Cb],True,(0,255,0),5)

    # The following two methods of finding inverse perspective transform are equivalent
    # Method 1
    H=cv2.getPerspectiveTransform(Cb.astype(np.float32),Ca.astype(np.float32))
    # Method 2
    # H1=cv2.getPerspectiveTransform(pt1,pt2)
    # H=np.linalg.inv(PT2)

    h,w=img.shape[:-1]
    warped = cv2.warpPerspective(img,H,(w,h))

    patch_A = img[Ca[0,1]:Ca[3,1],Ca[0,0]:Ca[1,0],:]
    patch_B = warped[Ca[0,1]:Ca[3,1],Ca[0,0]:Ca[1,0],:]

    H_4pt=Cb-Ca
    # check
    # mat=cv2.getPerspectiveTransform((H_4pt+Ca).astype(np.float32),Ca.astype(np.float32))
    # warpcheck = cv2.warpPerspective(img,mat,(w,h))
    # cv2.imshow('original',img)
    # cv2.imshow('warpcheck',warpcheck)
    # cv2.imshow('warped',warped)
    # cv2.waitKey(0)
    # ------
    # print(H_4pt.shape)
    print(patch_B.shape)
    if(H_4pt.shape[0]!=4):
        return False,_, _, _,_,_

    return True, patch_A, patch_B, H_4pt,Ca,Cb

if __name__=="__main__":
    ImagesPath=r"./../Data/Train"
    PatchAPath=r"./../Data/PatchA"
    PatchBPath=r"./../Data/PatchB"
    H4Path=r"./../Data/H4"

    if not os.path.exists(ImagesPath):
        print("The images path does not exist!")
        exit
    if not os.path.exists(PatchAPath):
        os.makedirs(PatchAPath)

    if not os.path.exists(PatchBPath):
        os.makedirs(PatchBPath)

    if not os.path.exists(H4Path):
        os.makedirs(H4Path)

    image_names=[]
    for dir,subdir,files in os.walk(ImagesPath):
        for file in files:
            image_names.append(file)
    count=0
    n_samples=5001
    H_4pt_array=[]
    Ca_array=[]
    Cb_array=[]

    while(count<n_samples):
        retval, patch_A, patch_B, H_4pt, Ca,Cb=GeneratePatches(ImagesPath,image_names)
        # cv2.imshow('patch_A',patch_A)
        # cv2.imshow('patch_B',patch_B)
        # cv2.waitKey()
        
        if(retval):
            cv2.imwrite(PatchAPath+os.sep+str(count)+'.jpg',patch_A)
            cv2.imwrite(PatchBPath+os.sep+str(count)+'.jpg',patch_B)
            H_4pt_array.append(H_4pt)
            Ca_array.append(Ca)
            Cb_array.append(Cb)
            count+=1
        else:
            print("Exception found in an image!")
    H_4pt_array=np.array(H_4pt_array)
    Ca_array=np.array(Ca_array)

    Cb_array=np.array(Cb_array)

    np.save(H4Path+os.sep+"H4.npy",H_4pt_array)
    np.save(H4Path+os.sep+"Ca.npy",Ca_array)

    np.save(H4Path+os.sep+"Cb.npy",Cb_array)

