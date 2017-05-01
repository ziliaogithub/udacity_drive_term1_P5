---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image10]: https://raw.githubusercontent.com/yashim/udacity_drive_term1_P5/master/non-vehicles/GTI/image19.png "Nonvehicle"
[image11]: https://raw.githubusercontent.com/yashim/udacity_drive_term1_P5/master/non-vehicles/GTI/image20.png "Nonvehicle"
[image20]: https://raw.githubusercontent.com/yashim/udacity_drive_term1_P5/master/vehicles/GTI_Far/image0958.png "Vehicle"
[image21]: https://raw.githubusercontent.com/yashim/udacity_drive_term1_P5/master/vehicles/GTI_Far/image0965.png "Vehicle"
[image2]: https://raw.githubusercontent.com/yashim/udacity_drive_term1_P5/master/output_images/hog.png "HOG visualization"
[image3]: ./examples/sliding_windows.jpg
[image4]: https://raw.githubusercontent.com/yashim/udacity_drive_term1_P5/master/output_images/result.png "Testing on test images"
[image5]: https://raw.githubusercontent.com/yashim/udacity_drive_term1_P5/master/output_images/heat_on_single_frame.png "Reducing overlaping boxes with heat"

[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png


[video1]: https://raw.githubusercontent.com/yashim/udacity_drive_term1_P4/master/with_lane.mp4 "Test video"
[video2]: https://raw.githubusercontent.com/yashim/udacity_drive_term1_P4/master/with_lane.mp4 "Result video"

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines #30 through #48 of the file called `training.py`. The method `extract_features` was taken from the lesson. 

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image10] ![alt text][image11]

![alt text][image20] ![alt text][image21]


I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` and `RGB` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

Below is output of 'training.py':
```
94.28 Seconds to extract features
Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 6156
My SVC predicts:  [ 0.  0.  1.  1.  0.  0.  0.  0.  1.  1.]
For these 10 labels:  [ 0.  0.  1.  1.  0.  0.  0.  0.  1.  1.]
0.00358 Seconds to predict 10 labels with SVC
```

#### 2. Explain how you settled on your final choice of HOG parameters.

I used HOG features, color features, color of histogram features. Below is the parameters to train my classifier, your can find corresponging code in `training.py` lines #13-15.
```
color_space='YCrCb'
orient = 9
pix_per_cell=8
cell_per_block=2
hog_channel='ALL'
spatial_size = (16, 16)
hist_bins=32
spatial_feat=True
hist_feat=True
hog_feat=True
```


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using LinearSVC class from sklearn.svm library.
The features have been extracted from images containing the vehicles and without them. The data was scaled, randomazed, splitted on two set: `train` and `test` (`training.py` line #53-#64). The vector and parameters are saved with help of `pickle` library. The whole process of training is presented in `training.py`

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scale to search and how much to overlap windows?

I used the subsampling algorithm presented in the class, this method allows extracting the Hog features only once. The code function `find_cars` are defined in `lesson_functions.py` file. This function is able to do both: extract features and make predictions. After the feature extracted then it can be sub-sampled in different windows. I have tried different combination of `scale` and `cells_per_step` (`pipeline.py` line #77 and `lesson_functions.py` line #283)
For example, below are timings on my computer.
```
---
scale=1.5
cells_per_step = 1 
[Finished in 41.782s]
(The most accurate option)
---
scale=1.5
cells_per_step = 2 
[Finished in 16.514s]
(Detection isn't accurate, too small patches)
---
scale=2
cells_per_step = 2 
[Finished in 9.927s]
(Detection isn't accurate, too small patches)
---
scale=2
cells_per_step = 1 
[Finished in 21.56s]
(No cars detected on test_video)
---
scale=3
cells_per_step = 1 
[Finished in 21.56s]
(Many false detection)

6)
scale=0.5
cells_per_step = 2 
[Finished in 158.861s]
Many false detection, and detects small parts of the car
```
For the final video `project_video.mp4` I choose `scale=1.5` and `cells_per_step = 2` because that configuration has shown better resulta on test vidoe. 

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]

For the better performance, I used sub-sampling algorithm, this approach allows extracting the Hog features only once and then the feature will be sub-sampled.
In order to run classifier only on part of the image which usually contains the cars, I did sub-sampling only on the slice of image which starts with y_start = 400 and ends with y_stop = 656 (all values among x axe).

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Below is a results of my algorithm processing of 'test_video.mp4' (click to the image below to open the video):

<a href="http://www.youtube.com/watch?feature=player_embedded&v=f3vBn_96vf8
" target="_blank"><img src="http://img.youtube.com/vi/f3vBn_96vf8/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="480" height="360" border="10" /></a>

And below is a results of processing of 'project_video.mp4' (click to the image below to open the video):

<a href="http://www.youtube.com/watch?feature=player_embedded&v=NlB9addIhBI
" target="_blank"><img src="http://img.youtube.com/vi/NlB9addIhBI/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="480" height="360" border="10" /></a>

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Here is the frame with multiple overlapping detections on the same car. The number of windows are reduced with `heat` function presented in `heat.py`:

![alt text][image5]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
Probably HOG will not work in real time, because it takes about half an hour to detectt car in the less than 1 minute video. Recently I tried the YOLO detector, which is based on Fast R-CNN (subcategory of conv neural network) and it can do better job in real time.
You can see from the output video the car detection is works quite good, but defenetly can be better. There are some frames in which the white car was not detected. For the detector it is hard to determine overlaping cars as two different cars.

