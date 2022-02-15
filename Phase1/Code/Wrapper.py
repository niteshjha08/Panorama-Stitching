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
import numpy as np
import cv2
import argparse
import os
from copy import deepcopy
# Add any python libraries here

def read_imgs(path):
	img_files = os.listdir(path)
	img_files.sort()
	img_arr = []
	for file in img_files:
		## Read image and append
		img = cv2.imread(f'{path}/{file}')
		img_arr.append(img)
	img_arr = np.array(img_arr)
	return img_arr

def get_corners(img_arr):
	corners_arr = []
	for img in img_arr:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		corners = cv2.cornerHarris(img, 2, 3, 0.04)
		corners_arr.append(corners)
	corners_arr = np.array(corners_arr)
	return corners_arr

def get_anms(img_arr, num_features):
	best_features_arr = []
	for img in img_arr:
		## First perform corner detection <I am using values found in example>
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		arr = cv2.goodFeaturesToTrack(img, num_features, 0.01, 10)
		best_features_arr.append(arr.astype(int))
	return best_features_arr

def get_feature_descriptors(img_arr, best_features, out_path, save_fig):
	fdescriptors_arr = []
	for i in range(len(img_arr)):
		img = img_arr[i]
		img_shape = img.shape
		f1 = []
		count = 0
		bf = best_features[i]
		for j in range(len(bf)):			
			[c,r] = bf[j][0]
			if c-20>=0 and c+20<=img_shape[0]-1 and r-20>=0 and r+20<=img_shape[1]-1:
				feature = img[r-20:r+20,c-20:c+20]
				feature = cv2.GaussianBlur(feature,(5,5),cv2.BORDER_DEFAULT)
				if i == 0 and count == 0 and save_fig:
					cv2.imwrite(f'{out_path}/FD_4141_{j}.jpg', feature)
				feature = cv2.resize(feature,(8,8))
				if i == 0 and count == 0 and save_fig:
					cv2.imwrite(f'{out_path}/FD_88_{j}.jpg', feature)
				feature = np.reshape(feature, (64,1,3))
				f1.append(feature)
				count+=1
		fdescriptors_arr.append(f1)
	return fdescriptors_arr


def main():
	# Add any Command Line arguments here
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--NumFeatures','-n', default=100, type=int, help='Number of best features to extract from each image, Default:100')
	Parser.add_argument('--path', '-p', type=str, help='Path where input data is', default="../Data/Train/Set1")
	Parser.add_argument('--save_fig', '-s', type=bool, help='True if you want to save outputs', default=True)
	Args = Parser.parse_args()
	NumFeatures = Args.NumFeatures
	in_path=Args.path
	out_path = '../Data/Outputs'
	if not os.path.isdir(out_path):
		os.makedirs(out_path)
	out_path = f'{out_path}/{in_path.split("/")[-1]}'
	if not os.path.isdir(out_path):
		os.makedirs(out_path)
	"""
	Read a set of images for Panorama stitching
	"""
	img_arr = read_imgs(in_path)
	"""
	Corner Detection
	Save Corner detection output as corners.png
	"""
	corners_arr = get_corners(img_arr)
	img_arr2 = deepcopy(img_arr)
	for i in range(len(img_arr2)):
		img = img_arr2[i]
		corners = corners_arr[i]
		for r in range(len(corners)):
			for c in range(len(corners[i])):
				if corners_arr[i][r][c] > 0.0:
					img[r,c,:] = [0,0,255]
		if Args.save_fig:
			cv2.imwrite(f'{out_path}/corners_{i+1}.jpg', img)

	"""
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""
	img_arr3 = deepcopy(img_arr)
	best_corners_arr = get_anms(img_arr, NumFeatures)
	for i in range(len(img_arr3)):
		img = img_arr3[i]
		best_c = best_corners_arr[i]
		for j in range(len(best_c)):
			[c,r] = best_c[j][0]
			img[r,c,:] = [0,0,255]
		if Args.save_fig:
			cv2.imwrite(f'{out_path}/anms_{i+1}.jpg', img)

	"""
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""
	feature_descriptors_arr = get_feature_descriptors(img_arr, best_corners_arr, out_path, Args.save_fig)
	print(np.shape(feature_descriptors_arr[0]),np.shape(feature_descriptors_arr[1]),np.shape(feature_descriptors_arr[2]))
	"""
	Feature Matching
	Save Feature Matching output as matching.png
	"""

	"""
	Refine: RANSAC, Estimate Homography
	"""


	"""
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""


if __name__ == '__main__':
	main()

