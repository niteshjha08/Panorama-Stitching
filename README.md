# Panorama-Stitching
Use traditional computer vision techniques and deep learning approach to generate panoramic image from multiple images
#### Phase 1
run using `python3 Wrapper.py`

#### Phase 2
Train: Start the training for supervised or unsupervised model by running `python3 Train.py --ModelType` where ModelType will be either Sup(for supervised) or Unsup(for unsupervised).

Test: Test the models by running `python3 Test.py --ModelType` 

## Pipeline

### 1. Corner detection
The features in images are found by detecting corners in images, and rejecting corners below a certain cornermetric

<p float="left">
	<img src="https://github.com/niteshjha08/Panorama-Stitching/blob/main/media/corners_1.jpg" width="400" />
	<img src="https://github.com/niteshjha08/Panorama-Stitching/blob/main/media/corners_2.jpg" width="400" />
</p>

### 2. To obtain corners uniformly distributed corners in the image (to avoid warping in blending), Adaptive Non-Maximal Suppression is done


<p float="left">
	<img src="https://github.com/niteshjha08/Panorama-Stitching/blob/main/media/anms_1.jpg" width="400" />
	<img src="https://github.com/niteshjha08/Panorama-Stitching/blob/main/media/anms_2.jpg" width="400" />
</p>

### 3. Feature matching
Feature descriptors are calculated around each of these corners (40x40), blurred and downsampled to 8x8 to form feature descriptor 64x1. Then feature matching is performed between all corner points in the two images, and top 2 matches are tested using Lowe's test ratio.


<p float="left">
	<img src="https://github.com/niteshjha08/Panorama-Stitching/blob/main/media/feature_matching0.jpg" width="800" />
</p> 

### 4. RANSAC
To eliminate wrong matches, RANSAC is performed to find out best points for finding homography between images

<p float="left">
	<img src="https://github.com/niteshjha08/Panorama-Stitching/blob/main/media/ransac0.jpg" width="800" />
</p>

### 5. Warping
Then, the two images are warped and blended to give the panorama image
<p float="left">
	<img src="https://github.com/niteshjha08/Panorama-Stitching/blob/main/media/stitched0.jpg" width="800" />
</p>
