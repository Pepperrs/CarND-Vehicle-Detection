# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./writeup/car.png
[image2]: ./writeup/noncar.png
[image3]: ./writeup/boxes.jpg
[image4]: ./writeup/boxes_around_cars.jpg
[image5]: ./writeup/heatmap.jpg
[image6]: ./writeup/heatmap_cars.jpg
[image7]: ./writeup/test1.jpg
[video1]: ./project_video.mp4

-----

### README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.
You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in cars.py in the function `generate_feature_map` which for the HOG classifier uses the function `get_hog_features`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]

The function `extract_features` then checks each image of both sets for HOG, spatial and histogram features, which are then scaled using `sklearn's` `StandardScaler`.
I ended up with the following parameters, since they provided the best results
* `color_space = "YUV"`
* `spatial_size = (32, 32)`
* `hist_bins = 32`
* `orientation = 7`
* `pix_per_cell = 16`
* `cell_per_block = 2`
* `hog_channel = "ALL"`

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and these seemed to give the most stable and correct results.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features and color features.

In the function `train_feature_map` I trained a linear SVM on histogram, spatial bins and HOG features using `sklearn's` `LinearSVC`.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to place search windows of different sizes across the lower portion of the image, only detecting vehicles on the road at a scale that would be appropriate to their possible distance.
This means that vehicles far away are only looked for close to the horizon, however larger vehicles are looked for on the whole visible road.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. I chose to apply heatmap labelung to reduce false positives. Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./writeup/project_Final.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from an image and the bounding boxes then overlaid on image:

Original image:

![alt text][image7]

Heatmap:

![alt text][image6]

Resulting boxes:

![alt text][image4]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The detection algorithm still does not work good enough to be used in an actual car. It fails under changes in light, for example when entering a tunnel or when driving at night. Also it is dependent on the dataset including all cars it will ever face. If the classifier would encounter a van or a truck, it would not detect it propperly. Also it could be mislead by bumperstickers or mirrorsm as it has no sanity check function.
