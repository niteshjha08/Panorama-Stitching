#!/usr/bin/evn python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano: Phase 1 Starter Code

Author(s): 
Chahat Deep Singh (chahat@terpmail.umd.edu) 
PhD Student in Computer Science,
University of Maryland, College Park

Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

# Code starts here:
from sys import flags
import numpy as np
import cv2
import argparse
import os
from copy import deepcopy
# Add any python libraries here
class Img:
    def __init__(self):
        self.name = ''
        self.grey = []
        self.rgb = []
        self.corners = []
        self. anms = []
        self.fd = []
        self.fdi = []
        self.kp = []
        self.temp_kp = []
        self.r_offset = 0 ## positive for offset to right, negative for left
        self.c_offset = 0 ## positive for top, negative for bottom

def read_imgs(path):
    img_files = os.listdir(path)
    img_files.sort()
    img_arr = []
    for f in img_files:
        ## Read image and append
        img = Img()
        img.rgb = cv2.imread(f'{path}/{f}')
        img.grey = cv2.cvtColor(img.rgb, cv2.COLOR_BGR2GRAY)
        img.grey = cv2.GaussianBlur(img.grey,(3,3),cv2.BORDER_DEFAULT)
        img_arr.append(img)
    return img_arr

def get_corners(img_arr):
    for img in img_arr:
        img.corners = cv2.cornerHarris(img.grey, 2, 3, 0.04)
    return img_arr

def get_anms(img_arr, num_features):
    for img in img_arr:
        ## First perform corner detection <I am using values found in example>
        img.anms = cv2.goodFeaturesToTrack(img.grey, num_features, 0.01, 10)
        img.anms = img.anms.astype('int32')
        rgb_anms = deepcopy(img.rgb)
        best_c = img.anms
        for j in range(len(best_c)):
            [c,r] = best_c[j][0]
            img_new2 = cv2.circle(rgb_anms, (c,r), 3, (0,0,255), 1)
        anms_new = []
        for i in range(len(img.anms)):
            [c,r] = img.anms[i,0]
            ## Eliminating wrong corners that are result of warping
            if 0 < c < img.grey.shape[1] and 0 < r < img.grey.shape[0]:
                c_s, c_e = c-20, c+20
                r_s, r_e = r-20, r+20
                if img.grey[r_s:r_e,c:c_e].all() == 0 or img.grey[r_s:r_e,c-1:c_e].all() == 0 or img.grey[r_s:r_e,c+1:c_e].all() == 0:
                    pass
                elif img.grey[r_s:r_e,c_s:c].all() == 0 or img.grey[r_s:r_e,c_s:c-1].all() == 0 or img.grey[r_s:r_e,c_s:c+1].all() == 0:
                    pass
                elif img.grey[r:r_e,c_s:c_e].all() == 0 or img.grey[r-1:r_e,c_s:c_e].all() == 0 or img.grey[r+1:r_e,c_s:c_e].all() == 0:
                    pass
                elif img.grey[r_s:r,c_s:c_e].all() == 0 or img.grey[r_s:r-1,c_s:c_e].all() == 0 or img.grey[r_s:r+1,c_s:c_e].all() == 0:
                    pass
                else:
                    anms_new.append(img.anms[i])
        img.anms = np.array(anms_new, dtype=int)
    return img_arr

def get_feature_descriptors(img_arr, fd_count, patch_size=41):
    offset = patch_size//2
    for i in range(len(img_arr)):
        img = img_arr[i]
        img_shape = img.grey.shape
        fd1 = []
        fdi1 = []
        for j in range(len(img.anms)):			
            [c,r] = img.anms[j][0]
            if c-offset>=0 and c+offset<=img_shape[0]-1 and r-offset>=0 and r+offset<=img_shape[1]-1:
                feature = img.grey[r-offset:r+offset,c-offset:c+offset]
                feature = cv2.GaussianBlur(feature,(5,5),cv2.BORDER_DEFAULT)
                if j < 20 and save_fig:
                    cv2.imwrite(f'{out_path}/FD_4141_{fd_count}.jpg', feature)
                feature = cv2.resize(feature,(8,8))
                if j < 20 and save_fig:
                    cv2.imwrite(f'{out_path}/FD_88_{fd_count}.jpg', feature)
                    fd_count+=1
                feature = np.reshape(feature, (64,1))
                feature = np.mean(feature, axis=1)
                feature = (feature - np.mean(feature,axis=0))/np.std(feature, axis = 0)
                fd1.append(feature)
                fdi1.append([c,r]) #saving colums, row
        img.fd = fd1
        img.fdi = fdi1
        img.kp = save_keypoints(img.fdi)
    return img_arr

def save_keypoints(fdi):
    kp_arr = []
    for i in range(len(fdi)):
        kp = cv2.KeyPoint(float(fdi[i][0]),float(fdi[i][1]),1.0)
        kp_arr.append(kp)
    return kp_arr

def get_feature_matching(img1, img2, threshold, count):
    '''
    Logic: 
    1. for each feature index in img_1, find SSD between it and all features indexes in img_2
    2. take the ratio of best and second best
    3. define a threshold and check if the ratio is less than the threshold
    4. if true, keep the feature pair
    '''
    fp_indexes = [] # [column, row] pair of the corresponding feature pair; shape: (n, 4)
    f1_idx_out = []
    f2_idx_out = []

    features1, f1_indexes = [], []
    features2, f2_indexes = [], []
    ## Match with lesser feature array
    if len(img1.fd) < len(img2.fd):
        features2 = img2.fd[:len(img1.fd)][:]
        f2_indexes = img2.fdi[:len(img1.fdi)][:]
        features1 = img1.fd
        f1_indexes = img1.fdi
    if len(img1.fd) > len(img2.fd):
        features1 = img1.fd[:len(img2.fd)][:]
        f1_indexes = img1.fdi[:len(img2.fdi)][:]
        features2 = img2.fd
        f2_indexes = img2.fdi
    ## take keypoints of corresponding features
    kp1 = img1.kp[:len(features1)]
    kp2 = img2.kp[:len(features2)]

    ## find best two matching points
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(np.asarray(features1,np.float32),np.asarray(features2,np.float32),2)
    ## matches is an array of cv2 DMatch object
    ## each match has distance, trainIdx(feature2), queryIdx(feature1), imgIdx(0 default)
    good = []
    original_threshold = threshold
    while len(good)<70:
        for m,n in matches: # m is best matched feature and n is second best
            if m.distance/n.distance <= threshold: # check if the ratio is good enough
                good.append([m])
                # save the indexes
                f1_c, f1_r = f1_indexes[m.queryIdx]
                f2_c, f2_r = f2_indexes[m.trainIdx]
                indexes = [f1_c, f1_r, f2_c, f2_r] 
                fp_indexes.append(indexes)
                f1_idx_out.append([f1_c,f1_r])
                f2_idx_out.append([f2_c,f2_r])
        threshold += 0.1
    threshold = original_threshold
    matched_image = cv2.drawMatchesKnn(img1.rgb, kp1, img2.rgb, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    if save_fig:
        cv2.imwrite(f'{out_path}/feature_matching{count}.jpg', matched_image)
    cv2.imshow("feature_matching", matched_image)
    if cv2.waitKey(0)==ord('q'):
        cv2.destroyAllWindows()
    
    return f1_idx_out, f2_idx_out, good

def performRANSAC(f1_idx, f2_idx, good, inlier_tolerance, num_iter=100):
    ## destination -- main image (img1); source -- changing image (img2)
    inlier_idx = []
    good_new = []
    i = 0
    # while (len(inlier_idx)<80) or (i<num_iter):
    vals = []
    original_tolerance = inlier_tolerance
    while len(inlier_idx)<10:
        for _ in range(num_iter):
            ## choose 4 random pairs
            pairs = np.random.randint(0,len(f1_idx),size=4)
            destination_features = np.float32([f1_idx[p] for p in pairs])
            source_features = np.float32([f2_idx[p] for p in pairs])
            h_estimate = cv2.getPerspectiveTransform(source_features, destination_features)

            for p in pairs:
                Hpi = np.dot(h_estimate, np.array([f1_idx[p][0], f1_idx[p][1], 1]))
                if Hpi[2] != 0:
                    Hpi_x = Hpi[0]/Hpi[2]
                    Hpi_y = Hpi[1]/Hpi[2]
                else:
                    Hpi_x = Hpi[0]/0.000001
                    Hpi_y = Hpi[1]/0.000001
                Hpi = np.array([Hpi_x, Hpi_y])
                ppi = f2_idx[p]
                ssd = np.linalg.norm(ppi-Hpi)
                vals.append(ssd)
                if ssd > 1.0 and ssd < inlier_tolerance:
                    # print(ssd)
                    if f1_idx[p] not in inlier_idx and f2_idx[p] not in inlier_idx:
                        inlier_idx.append(f1_idx[p])
                        inlier_idx.append(f2_idx[p])
                        good_new.append(good[p])
        diff = abs(inlier_tolerance-min(vals))+30.0
        inlier_tolerance += diff
        # i += 1
        # if len(inlier_idx)==len(f1_idx):
    inlier_tolerance = original_tolerance
    destination_features = np.float32([inlier_idx[i] for i in range(0,len(inlier_idx),2)])
    source_features = np.float32([inlier_idx[i]for i in range(1,len(inlier_idx),2)])
    homography, _ = cv2.findHomography(source_features, destination_features, method=0, ransacReprojThreshold=0.0)
    return np.asarray(inlier_idx), good_new, homography

def warpTwoImages(img1, img2, H, r_offset, c_offset):
    '''warp, blend and stitch img2 to img1 with homograph H'''
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin+r_offset, ymax-ymin+c_offset))
    r_s = max(t[1]-r_offset,0)
    r_e = h1+r_s
    c_s = max(t[0]-c_offset, 0)
    c_e = w1+t[0]
    if c_s-c_e != img1.shape[1]:
        c_s = t[0]
        c_e = w1+t[0]
    if r_s-r_e != img1.shape[0]:
        r_s = t[1]
        r_e = h1+t[1]
    r_i = 0
    for r in range(r_s, r_e):
        c_i = 0
        for c in range(c_s, c_e):
            if img1[r_i,c_i,:].all() != 0:
                result[r,c,:] = img1[r_i,c_i,:]
            c_i += 1
        r_i += 1
    # result[t[1]-r_offset//2:h1+t[1]-r_offset//2, t[0]-c_offset:w1+t[0]-c_offset] = img1
    offset_y = t[0]-c_offset
    offset_x = t[1]-r_offset
    return result, offset_x, offset_y

def stitching_pipeline(img1, img2, count):
    ## img1 features(destination -- the one we want to stitch to), img2 features(Source --  the one we want to change to match destination)
    f1_idx, f2_idx, good = get_feature_matching(img1, img2, threshold=FM_THRESHOLD, count=count)
    print('Feature Matching output shapes:',np.shape(f1_idx))
    
    """
    Refine: RANSAC, Estimate Homography
    """
    ransac_idx, good_new, homography = performRANSAC(f1_idx, f2_idx, good, RANSAC_THRESHOLD)
    print('RANSAC output shapes:',np.shape(ransac_idx))

    ## plot the result
    matched_image = cv2.drawMatchesKnn(img1.rgb, img1.kp, img2.rgb, img2.kp, good_new, None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("ransac", matched_image)
    if cv2.waitKey(0)==ord('q'):
        cv2.destroyAllWindows()
    if save_fig:
        cv2.imwrite(f'{out_path}/ransac{count}.jpg', matched_image)
    
    """
    Image Warping + Blending
    Save Panorama output as mypano.png
    """
    warped_img, r_offset, c_offset = warpTwoImages(img1.rgb, img2.rgb, homography, img1.r_offset, img1.c_offset)
    
    print(r_offset, c_offset)
    cv2.imshow("warped_img", warped_img)
    if cv2.waitKey(0)==ord('q'):
        cv2.destroyAllWindows()
    if save_fig:
        cv2.imwrite(f'{out_path}/stitched{count}.jpg', warped_img)
    """
    Processing the Stitched image
    """
    stitched_img = Img()
    stitched_img.r_offset += r_offset
    stitched_img.c_offset += c_offset
    stitched_img.rgb = warped_img
    stitched_img.grey = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    ## Corners calculation
    [stitched_img] = get_corners([stitched_img])
    print("Stitched Corners shapes:",np.shape(stitched_img.corners))

    if save_fig:
        rgb_corners = deepcopy(stitched_img.rgb)
        corners_arr = stitched_img.corners
        corners_arr[corners_arr<0] = 0
        mean = np.mean(corners_arr)
        for r in range(len(corners_arr)):
            for c in range(len(corners_arr[0])):
                if corners_arr[r][c] > mean:
                    img_new = cv2.circle(rgb_corners, (c,r), 1, (0,0,255), 1)
        cv2.imwrite(f'{out_path}/stitched_corners_{count}.jpg', img_new)

    ## ANMS Calculation
    [stitched_img] = get_anms([stitched_img], NumFeatures)
    stitched_img.anms = stitched_img.anms[10:]
    print("Stitched ANMS shapes:",np.shape(stitched_img.anms))
    
    rgb_anms = deepcopy(stitched_img.rgb)
    best_c = stitched_img.anms
    for j in range(len(best_c)):
        [c,r] = best_c[j][0]
        img_new2 = cv2.circle(rgb_anms, (c,r), 3, (0,0,255), 1)
    if save_fig:
        cv2.imwrite(f'{out_path}/stitched_anms_{count}.jpg', img_new)
    cv2.imshow("anms", img_new2)
    if cv2.waitKey(0)==ord('q'):
        cv2.destroyAllWindows()
    [stitched_img] = get_feature_descriptors([stitched_img], fd_count)
    return stitched_img

def main():
    # Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--NumFeatures','-n', default=300, type=int, help='Number of best features to extract from each image, Default:100')
    Parser.add_argument('--path', '-p', type=str, help='Path where input data is', default="../Data/Train/Set1")
    Parser.add_argument('--save_fig', '-s', type=bool, help='True if you want to save outputs', default=False)
    Args = Parser.parse_args()
    
    global out_path, FM_THRESHOLD, RANSAC_THRESHOLD, NumFeatures, save_fig, fd_count
    ## Hyper params
    fd_count = 0
    NumFeatures = Args.NumFeatures
    save_fig = Args.save_fig
    # in_path=Args.path
    in_path="../Data/Train/Set2"
    out_path = '../Data/Outputs'
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_path = f'{out_path}/{in_path.split("/")[-1]}'
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    FM_THRESHOLD = 0.80
    RANSAC_THRESHOLD = 100.0
    """
    Read a set of images for Panorama stitching
    """
    img_arr = read_imgs(in_path) ## gray scale images for processing, rgb images for visualization
    LEFT = len(img_arr) # total number of images left to be stitched
    """
    Corner Detection
    Save Corner detection output as corners.png
    """
    img_arr = get_corners(img_arr)
    print("Corners output shapes:",np.shape(img_arr[0].corners),np.shape(img_arr[1].corners),np.shape(img_arr[2].corners))
    ## Save output images
    ## corners
    for i in range(len(img_arr)):
        img = img_arr[i]
        rgb_corners = deepcopy(img.rgb)
        corners_arr = img.corners
        corners_arr[corners_arr<0] = 0
        mean = np.mean(corners_arr)
        for r in range(len(corners_arr)):
            for c in range(len(corners_arr[0])):
                if corners_arr[r][c] > mean:
                    img_new = cv2.circle(rgb_corners, (c,r), 1, (0,0,255), 1)
        if save_fig:
            cv2.imwrite(f'{out_path}/corners_{i+1}.jpg', img_new)
    """
    Perform ANMS: Adaptive Non-Maximal Suppression
    Save ANMS output as anms.png
    """
    img_arr = get_anms(img_arr, NumFeatures)
    ## Save output images
    ## anms
    for i in range(len(img_arr)):
        img = img_arr[i]
        rgb_anms = deepcopy(img.rgb)
        best_c = img.anms
        for j in range(len(best_c)):
            [c,r] = best_c[j][0]
            img_new2 = cv2.circle(rgb_anms, (c,r), 3, (0,0,255), 1)
        if save_fig:
            cv2.imwrite(f'{out_path}/anms_{i+1}.jpg', img_new2)
    print("AMNS output shapes:",np.shape(img_arr[0].anms),np.shape(img_arr[1].anms),np.shape(img_arr[2].anms))
    """
    Feature Descriptors
    Save Feature Descriptor output as FD.png
    """
    img_arr = get_feature_descriptors(img_arr, fd_count)
    print("Feature Descriptors output shapes:",np.shape(img_arr[0].fd),np.shape(img_arr[1].fd),np.shape(img_arr[2].fd))
    print("Feature Descriptors Index output shapes:",np.shape(img_arr[0].fdi),np.shape(img_arr[1].fdi),np.shape(img_arr[2].fdi))
    """
    Feature Matching
    Save Feature Matching output as matching.png
    """
    anchor_img = LEFT//2
    count = 0

    for i in range(anchor_img,0,-1):
        if i == anchor_img:
            img1 = img_arr[i] ## the one we want to stitch to
        img2 = img_arr[i-1] ## the one we want to warp
        stitched_img = stitching_pipeline(img1, img2, count)
        count+=1
        img1 = stitched_img
    
    img1 = stitched_img ## the one we want to stitch to
    for i in range(anchor_img, anchor_img+1):
        img2 = img_arr[i+1] ## the one we want to warp
        stitched_img = stitching_pipeline(img1, img2, count)
        count += 1
        img1 = stitched_img
    cv2.imshow("final",stitched_img.rgb)
    if cv2.waitKey(0)==ord('q'):
        cv2.destroyAllWindows()
    if save_fig:
        cv2.imwrite(f'{out_path}/final_img.jpg', stitched_img.rgb)
if __name__ == '__main__':
    main()


## How to have interactive imshow
# k = cv2.waitKey(0)
# if k == ord('a'):
# 	cv2.destroyAllWindows()