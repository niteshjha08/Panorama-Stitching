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

def read_imgs(path):
	img_files = os.listdir(path)
	img_files.sort()
	img_arr = []
	for f in img_files:
		## Read image and append
		img = Img()
		img.rgb = cv2.imread(f'{path}/{f}')
		img.grey = cv2.cvtColor(img.rgb, cv2.COLOR_BGR2GRAY)
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
		img.anms = img.anms.astype(int)
	return img_arr

def get_feature_descriptors(img_arr, out_path, patch_size=41, save_fig=False):
	fd_arr = []
	fd_indexes = []
	offset = patch_size//2
	count = 0
	for i in range(len(img_arr)):
		img = img_arr[i]
		img_shape = img.grey.shape
		fd1 = []
		fdi1 = []
		for j in range(len(img.anms)):			
			[c,r] = img.anms[j][0]
			if c-offset>=0 and c+offset<=img_shape[0]-1 and r-offset>=0 and r+offset<=img_shape[1]-1:
				feature = img.grey[r-offset:r+offset,c-offset:c+offset]
				feature = cv2.GaussianBlur(feature,(7,7),cv2.BORDER_DEFAULT)
				if i == 0 and count == 0 and save_fig:
					cv2.imwrite(f'{out_path}/FD_4141_{j}.jpg', feature)
				feature = cv2.resize(feature,(8,8))
				if i == 0 and count == 0 and save_fig:
					cv2.imwrite(f'{out_path}/FD_88_{j}.jpg', feature)
					count+=1
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

def get_feature_matching(img1, img2, out_path, threshold, save_fig=False):
	'''
	Logic: 
	1. for each feature index in img_1, find SSD between it and all features indexes in img_2
	2. take the ratio of best and second best
	3. define a threshold and check if the ratio is less than the threshold
	4. if true, keep the feature pair
	'''
	feature_pairs = [] #each one is an array of feature pair; shape: (n, 2, 62)
	fp_indexes = [] # [column, row] pair of the corresponding feature pair; shape: (n, 4)
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

	bf = cv2.BFMatcher()
	matches = bf.knnMatch(np.asarray(features1,np.float32),np.asarray(features2,np.float32),2)
	## matches is an array of cv2 DMatch object
	## each match has distance, trainIdx(feature2), queryIdx(feature1), imgIdx(0 default)
	good = []
	for m,n in matches: # m is best matched feature and n is second best
		if m.distance/n.distance <= threshold: # check is the ratio is good enough
			good.append([m])
			# save the indexes
			f1_c, f1_r = f1_indexes[m.queryIdx]
			f2_c, f2_r = f2_indexes[m.trainIdx]
			indexes = [f1_c, f1_r, f2_c, f2_r] 
			fp_indexes.append(indexes)
			# save the features
			f_arr = np.array([features1[m.queryIdx],features2[m.trainIdx]])
			feature_pairs.append(f_arr)

			# Sum of Squared Difference calculation
			# m_check = np.sum(np.square(features1[m.queryIdx] - features2[m.trainIdx]))
			# print(m.distance, np.sqrt(m_check), m_check)
	if save_fig:	

		matched_image = cv2.drawMatchesKnn(img1.rgb, kp1, img2.rgb, kp2, good, None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
		cv2.imwrite(f'{out_path}/feature_matching.jpg', matched_image)

	return feature_pairs, fp_indexes, good

def performRANSAC(feature_pairs, fp_indexes, good, inlier_tolerance, num_iter, save_fig=False):
	inlier_fp = []
	inlier_idx = []
	good_new = []
	for i in range(num_iter):
		## choose 4 random pairs
		pairs = np.random.randint(0,len(fp_indexes),size=4)
		source_features = np.float32([[[fp_indexes[p][0],fp_indexes[p][1]]] for p in pairs])
		destination_features = np.float32([[[fp_indexes[p][2],fp_indexes[p][3]]] for p in pairs])
		H, _ = cv2.findHomography(source_features, destination_features, method=0, ransacReprojThreshold=0.0)
		try:
			for p in pairs:
				Hpi = np.dot(H,np.array([[fp_indexes[p][0]],[fp_indexes[p][1]],[1]]))
				ppi = np.array([[fp_indexes[p][2]],[fp_indexes[p][3]],[1]])
				ssd = np.sqrt(np.sum(np.square(ppi-Hpi)))
				if ssd < inlier_tolerance:
					if fp_indexes[p] not in inlier_idx:
						inlier_fp.append(feature_pairs[p])
						inlier_idx.append([fp_indexes[p][:2]])
						inlier_idx.append([fp_indexes[p][2:]])
						good_new.append(good[p])
		except TypeError:
			pass

	source_features = np.float32([inlier_idx[i] for i in range(0,len(inlier_idx),2)])
	destination_features = np.float32([inlier_idx[i]for i in range(1,len(inlier_idx),2)])

	H, _ = cv2.findHomography(source_features, destination_features, method=0, ransacReprojThreshold=0.0)
	return inlier_fp, np.asarray(inlier_idx), good_new, H

def main():
	# Add any Command Line arguments here
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--NumFeatures','-n', default=100, type=int, help='Number of best features to extract from each image, Default:100')
	Parser.add_argument('--path', '-p', type=str, help='Path where input data is', default="../Data/Train/Set1")
	Parser.add_argument('--save_fig', '-s', type=bool, help='True if you want to save outputs', default=False)
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
		if Args.save_fig:
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
		if Args.save_fig:
			cv2.imwrite(f'{out_path}/anms_{i+1}.jpg', img_new2)
	print("AMNS output shapes:",np.shape(img_arr[0].anms),np.shape(img_arr[1].anms),np.shape(img_arr[2].anms))
	"""
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""
	img_arr = get_feature_descriptors(img_arr, out_path, patch_size=41, save_fig=Args.save_fig)
	print("Feature Descriptors output shapes:",np.shape(img_arr[0].fd),np.shape(img_arr[1].fd),np.shape(img_arr[2].fd))
	print("Feature Descriptors Index output shapes:",np.shape(img_arr[0].fdi),np.shape(img_arr[1].fdi),np.shape(img_arr[2].fdi))
	"""
	Feature Matching
	Save Feature Matching output as matching.png
	"""
	matching_f12, matching_i12, good = get_feature_matching(img_arr[0], img_arr[1], out_path, threshold=0.8, save_fig=Args.save_fig)
	print('Feature Matching output shapes:',np.shape(matching_f12),np.shape(matching_i12))
	"""
	Refine: RANSAC, Estimate Homography
	"""
	ransac_fp, ransac_idx, good_new, H = performRANSAC(matching_f12, matching_i12, good, 100, len(matching_f12), save_fig=Args.save_fig)
	print('RANSAC output shapes:',np.shape(ransac_fp),np.shape(ransac_idx))
	## plot the result
	matched_image = cv2.drawMatchesKnn(img_arr[0].rgb, img_arr[0].kp, img_arr[1].rgb, img_arr[1].kp, good_new, None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
	if Args.save_fig:
		cv2.imwrite(f'{out_path}/ransac.jpg', matched_image)
	check_distance = np.sum(np.sum(ransac_idx[::2]-ransac_idx[1::2], axis=1), axis=0)
	
	"""
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""
	if check_distance[0] < check_distance[1]:
		height = img_arr[0].rgb.shape[0] + img_arr[1].rgb.shape[0]
		wdith = img_arr[1].rgb.shape[1]
	else:
		height = img_arr[0].rgb.shape[0]
		wdith = img_arr[1].rgb.shape[1] + img_arr[1].rgb.shape[1]
	result = cv2.warpPerspective(img_arr[1].rgb, H, (img_arr[1].rgb.shape[1], img_arr[1].rgb.shape[0]))
	# width = img_arr[0].rgb.shape[1]
	# height = img_arr[0].rgb.shape[0] + result.shape[0]
	template = np.zeros((height,width,3), np.uint8)
	template[0:img_arr[0].rgb.shape[0],:] = img_arr[0].rgb
	template[img_arr[0].rgb.shape[0]:,:] = result
	# cv2.imshow('stitched', template)
	# cv2.imshow('img1', img_arr[0].rgb)
	# if cv2.waitKey(0) == ord('q'):
	# 	cv2.destroyAllWindows
if __name__ == '__main__':
	main()


## How to have interactive imshow
# k = cv2.waitKey(0)
# if k == ord('a'):
# 	cv2.destroyAllWindows()