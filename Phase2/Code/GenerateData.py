#!/usr/bin/python3
import cv2
import numpy as np
import random
import math
import os

# def GenerateData(ImagesPath, patch_size=200,rho=50):
#     image_names=[]
#     for dir,subdir,files in os.walk(ImagesPath):
#         for file in files:
#             image_names.append(file)
    
#     idx=random.randint(0,len(image_names))

#     path=ImagesPath + os.sep + image_names[idx]
#     img=cv2.imread(path)
#     M,N,_=img.shape


#     # select patch dimensions
#     Mp, Np = random.randint(50,M-2*rho), random.randint(50,N-2*rho)



#     # Top left corner of patch selected
#     print("M rand is between range: ",rho," to ",M-Mp-rho)
#     print("N rand is between range: ",rho," to ",N-Np-rho)
#     Ca1=np.array([random.randint(rho,M-Mp-rho),random.randint(rho,N-Np-rho)])
#     print(Ca1.shape)

#     Ca=np.array([[Ca1[1],Ca1[0]],
#                  [Ca1[1]+Np,Ca1[0]],
#                  [Ca1[1]+Np,Ca1[0]+Mp ],
#                  [Ca1[1],Ca1[0]+Mp]])

#     # Finding patch after random perturbations
#     Cb=np.zeros(Ca.shape,dtype=int)
#     print(Ca.shape)
#     for i in range(4):
#         rx=random.randint(-rho,rho)
#         ry=random.randint(-rho,rho)
#         Cb[i,0]=Ca[i,0] + ry
#         Cb[i,1]=Ca[i,1] + rx


#     print("Ca:",Ca)
#     cv2.rectangle(img,Ca[0],Ca[2],(0,0,255),5)

#     print("Cb:",Cb)
#     cv2.rectangle(img,Cb[0],Cb[2],(255,0,0),5)

#     cv2.imshow('patch',img)


#     cv2.waitKey(0)

def GeneratePatches(ImagesPath,image_names, patch_size=200,rho=50):

    idx=random.randint(0,len(image_names))

    path=ImagesPath + os.sep + image_names[idx]
    img=cv2.imread(path)  
    
    M,N,_=img.shape
    print(img.shape)
    print("image:",path)
    print(patch_size+rho+1)
    if((M<patch_size+2*rho+1) | (N<patch_size+2*rho+1)):
        return False,_,_,_

    # Top left corner of patch selected
    # print("M rand is between range: ",rho," to ",M-patch_size-rho) 
    # print("N rand is between range: ",rho," to ",N-patch_size-rho)
    Ca1=np.array([random.randint(rho,M-patch_size-rho),random.randint(rho,N-patch_size-rho)])

    Ca=np.array([[Ca1[1],Ca1[0]],
                 [Ca1[1]+patch_size,Ca1[0]],
                 [Ca1[1]+patch_size,Ca1[0]+patch_size ],
                 [Ca1[1],Ca1[0]+patch_size]])

    # Finding patch after random perturbations
    Cb=np.zeros(Ca.shape,dtype=int)

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

    return True, patch_A, patch_B, H_4pt

# def checkperspective():
#     img=cv2.imread("/home/nitesh/programming/CMSC733/P1/niteshj_p1/Phase2/Data/Train/6.jpg")
#     # print(img.shape)
#     pt1=np.array([[300,50],[400,50],[400,250],[300,250]])

#     pt2=np.array([[300,50],[400,20],[400,280],[300,250]])
#     imcopy=img.copy()
#     cv2.polylines(img,[pt1],True,(0,255,0),5)
#     cv2.polylines(img,[pt2],True,(0,255,255),5)

#     imcopy=img.copy()
#     pt1=np.float32(pt1)
#     pt2=np.float32(pt2)
    
#     # The following two methods of finding inverse perspective transform are equivalent
#     # Method 1
#     H=cv2.getPerspectiveTransform(pt2,pt1)
#     # Method 2
#     # H1=cv2.getPerspectiveTransform(pt1,pt2)
#     # H=np.linalg.inv(PT2)

#     warped = cv2.warpPerspective(img,H,img.shape[:-1])

#     pt1=pt1.astype(int)
#     # print(pt1)
#     patch_A = img[pt1[0,1]:pt1[3,1],pt1[0,0]:pt1[1,0],:]

#     patch_B = warped[pt1[0,1]:pt1[3,1],pt1[0,0]:pt1[1,0],:]
#     cv2.imshow('patchA:',patch_A)
#     cv2.imshow('img',img)
#     cv2.imshow('warped',warped)
#     cv2.imshow('patchB',patch_B)
#     H_4pt = pt2 - pt1
#     print(H_4pt)
#     cv2.waitKey()

if __name__=="__main__":
    ImagesPath=r"/home/nitesh/programming/CMSC733/P1/niteshj_p1/Phase2/Data/Train"
    PatchAPath=r"/home/nitesh/programming/CMSC733/P1/niteshj_p1/Phase2/Data/PatchA"
    PatchBPath=r"/home/nitesh/programming/CMSC733/P1/niteshj_p1/Phase2/Data/PatchB"
    if not os.path.exists(PatchAPath):
        os.makedirs(PatchAPath)

    if not os.path.exists(PatchBPath):
        os.makedirs(PatchBPath)

    image_names=[]
    for dir,subdir,files in os.walk(ImagesPath):
        for file in files:
            image_names.append(file)
    count=0
    n_samples=10
    while(count<n_samples):
        retval, patch_A, patch_B, H_4pt=GeneratePatches(ImagesPath,image_names)
        # cv2.imshow('patch_A',patch_A)
        # cv2.imshow('patch_B',patch_B)
        # cv2.waitKey()
        cv2.imwrite(PatchAPath+os.sep+str(count)+'.jpg',patch_A)
        cv2.imwrite(PatchBPath+os.sep+str(count)+'.jpg',patch_B)

        
        if(retval):
            count+=1
        else:
            print("Exceptional image!")
