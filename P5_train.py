import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import pickle
import sklearn.svm as svm
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.externals import joblib
from lesson_functions import *
from sklearn.model_selection import train_test_split

cars = glob.glob('vehicles/**/*.png')
notcars = glob.glob('non-vehicles/**/*.png')

# loading car images
car_image = []
for impath in cars:
    car_image.append(impath)

# loading non car images
notcar_image = []
for impath in notcars:
    notcar_image.append(impath)

car_image_count = len(car_image)
notcar_image_count = len(notcar_image)

print ('cars:', car_image_count)
print ('not cars:', notcar_image_count)


# helper function to extract features from files

def get_features(files, color_space='RGB', spatial_size=(32, 32),
                 hist_bins=32, orient=9,
                 pix_per_cell=8, cell_per_block=2, hog_channel=0,
                 spatial_feat=True, hist_feat=True, hog_feat=True):
    features = []
    for file in files:
        img = mpimg.imread(file)
        img_features = single_img_features(img, color_space=color_space, spatial_size=spatial_size,
                                           hist_bins=hist_bins, orient=orient,
                                           pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                           hog_channel=hog_channel,
                                           spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)

        features.append(img_features)
    return features


color_space = 'HLS'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
spatial_size = (16, 16)
hist_bins = 32
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
spatial_feat = True
hist_feat = True
hog_feat = True

t = time.time()
cars_train_feat = get_features(car_image, color_space, spatial_size, hist_bins, orient,
                               pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)
notcars_train_feat = get_features(notcar_image, color_space, spatial_size, hist_bins, orient,
                                  pix_per_cell, cell_per_block, hog_channel, spatial_feat, hist_feat, hog_feat)

t2 = time.time()
print(round(t2 - t, 2), 'Seconds to extract HOG,spatial and color features...')

# Create an array stack of feature vectors
X = np.vstack((cars_train_feat, notcars_train_feat)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(cars_train_feat)), np.zeros(len(notcars_train_feat))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

X_train, y_train = shuffle(X_train, y_train, random_state=42)

print('Using:', orient, 'orientations', pix_per_cell, 'pixels per cell and', cell_per_block, 'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
svc = LinearSVC()

# use of the rbf kernel improves the accuracy by about another percent,
# but increases the prediction time up to 1.7s(!) for 100 labels. Too slow.
# svc = svm.SVC(kernel='rbf')

# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample


# Save the data for easy access
pickle_file = 'ProcessedData.p'
print('Saving data to pickle file...')
try:
    with open(pickle_file, 'wb') as pfile:
        pickle.dump(
            {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            },
            pfile, pickle.HIGHEST_PROTOCOL)
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

print('Data cached in pickle file.')

pickle_file = 'ClassifierData.p'
print('Saving data to pickle file...')
try:
    with open(pickle_file, 'wb') as pfile:
        pickle.dump(
            {'svc': svc,
             'X_scaler': X_scaler,
             'color_space': color_space,
             'spatial_size': spatial_size,
             'hist_bins': hist_bins,
             'orient': orient,
             'pix_per_cell': pix_per_cell,
             'cell_per_block': cell_per_block,
             'hog_channel': hog_channel,
             'spatial_feat': spatial_feat,
             'hist_feat': hist_feat,
             'hog_feat': hog_feat
             },
            pfile, pickle.HIGHEST_PROTOCOL)
except Exception as e:
    print('Unable to save data to', pickle_file, ':', e)
    raise

print('Data cached in pickle file.')


