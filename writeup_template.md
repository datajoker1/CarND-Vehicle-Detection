# Vehicle Detection and Tracking


# Getting Started
* Clone the project and create directories `vehicles` and `non-vehicles`. 
* Download images from the links given below and put them into subfolders below `vehicles` and `non-vehicles`. 
* `P5.ipynb` Preview pipeline.
* `P5_train.py` trains an SVM to detect cars and non-cars. All classifier data is saved in  a pickle file.
* `P5_search.py` implements a sliding window search for cars

---
# Data Exploration
In total there are 8792 images of vehicles and 8968 images of non vehicles. 
Thus the data is slightly unbalanced with about 10% more non vehicle images than vehicle images.
Shown below is an example of each class (vehicle, non-vehicle) of the data set. The data set is explored in the notebook `P5.ipynb` 

![output_images][load_data]


# Histogram of Oriented Gradients (HOG)

## Extraction of HOG, color and spatial features
Each set was shuffled individually. The code for this step is contained in the first six cells of the IPython notebook `HOG_Classify.ipynb`. I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  

![output_images][explore]

##  Choice of parameters and channels for HOG
I experimented with a number of different combinations of color spaces and HOG parameters and trained  a linear SVM using different combinations of HOG features extracted from the color channels. For HLS color space the L-channel appears to be most important, followed by the S channel. I discarded RGB color space, for its undesirable properties under changing light conditions. 

## Training a linear SVM on the final choice of features

I trained a linear SVM using all channels of images converted to HLS space. I included spatial features color features as well as all three HLS channels, because using less than all three channels reduced the accuracy considerably. 
The final feature vector has a length of 6156 elements, most of which are HOG features. For color binning patches of `spatial_size=(16,16)` were generated and color histograms 
were implemented using `hist_bins=32` used. 
LinearSVM is better performance than the others.. (SVM, GaussianNB, DecisionTreeClassifier)


# Sliding Window Search

## Implementation of the sliding window search
In the file `P5_search.py` I  segmented the image into 4 partially overlapping zones with different sliding window sizes to account for different distances.
The window sizes are  240,180,120 and 70 pixels for each zone. Within each zone adjacent windows have an ovelap of 75%, as illustrated below. The search over all zones is implemented in the `search_all_scales(image)` function. Using even slightly less than 75% overlap resulted in an unacceptably large number of false negatives. 


## Search examples
The final classifier uses four scales and HOG features from all 3 channels of images in HLS space. The feature vector contains also  spatially binned color and histograms of color features 
False positives occured more frequently for `pixels_per_cell=8` compared to `pixels_per_cell=16`, but nevertheless produced better results when applied to the video. The false positives 
were filtered out by using a heatmap approach

In the file `P5_search.py` the class `BoundingBoxes` implements a FIFO queue that stores the bounding boxes of the last `n` frames. 
For every frame the (possbly empty) list of detected bounding boxes gets added to the beginning of the queue, while the oldest list of bounding boxes falls out. 
This queue is then used in the processing of the video and always contains the bounding boxes of the last `n=20` frames. 
![output_images][result]

---
# Discussion

## Problems / issues encountered and outlook

1. I started out with a linear SVM due to its fast evaluation. Nonlinear kernels such as `rbf` take not only longer to train, but also much longer to evaluate.
A way to improve speed would be to compute the HOG features only once for the entire region of interest and then select the right feature vectors, when the image is slid across. 

2. Some false positives still remain after heatmap filtering. This should be improvable by using more labeled data. 




