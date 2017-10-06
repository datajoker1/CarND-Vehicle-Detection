import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from lesson_functions import *
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import imageio

imageio.plugins.ffmpeg.download()
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from collections import deque
from scipy.ndimage.measurements import label
from IPython.core.debugger import Tracer;

debug_here = Tracer()
from tqdm import tqdm

# Load the classifier and parameters
data_file = 'ClassifierData.p'
with open(data_file, mode='rb') as f:
    data = pickle.load(f)

svc = data['svc']
X_scaler = data['X_scaler']
color_space = data['color_space']
spatial_size = data['spatial_size']
hist_bins = data['hist_bins']
orient = data['orient']
pix_per_cell = data['pix_per_cell']
cell_per_block = data['cell_per_block']
hog_channel = data['hog_channel']
spatial_feat = data['spatial_feat']
hist_feat = data['hist_feat']
hog_feat = data['hog_feat']

images = sorted(glob.glob('test_images/out*.png'))


def search_all_scales(image):
    hot_windows = []
    all_windows = []

    X_start_stop = [[None, None], [None, None], [None, None], [None, None]]
    w0, w1, w2, w3 = 240, 180, 120, 70
    o0, o1, o2, o3 = 0.75, 0.75, 0.75, 0.75
    XY_window = [(w0, w0), (w1, w1), (w2, w2), (w3, w3)]
    XY_overlap = [(o0, o0), (o1, o1), (o2, o2), (o3, o3)]
    yi0, yi1, yi2, yi3 = 380, 380, 395, 405
    Y_start_stop = [[yi0, yi0 + w0 / 2], [yi1, yi1 + w1 / 2], [yi2, yi2 + w2 / 2], [yi3, yi3 + w3 / 2]]

    for i in range(len(Y_start_stop)):
        windows = slide_window(image, x_start_stop=X_start_stop[i], y_start_stop=Y_start_stop[i],
                               xy_window=XY_window[i], xy_overlap=XY_overlap[i])

        all_windows += [windows]

        hot_windows += search_windows(image, windows, svc, X_scaler, color_space=color_space,
                                      spatial_size=spatial_size, hist_bins=hist_bins,
                                      orient=orient, pix_per_cell=pix_per_cell,
                                      cell_per_block=cell_per_block,
                                      hog_channel=hog_channel, spatial_feat=spatial_feat,
                                      hist_feat=hist_feat, hog_feat=hog_feat)

    return hot_windows, all_windows

# ## Video Pipeline
# Define a class to receive the characteristics of bounding box detections
class BoundingBoxes:
    def __init__(self, n=10):
        # length of queue to store data
        self.n = n
        # hot windows of the last n images
        self.recent_boxes = deque([], maxlen=n)
        # current boxes
        self.current_boxes = None
        self.allboxes = []

    def add_boxes(self):
        self.recent_boxes.appendleft(self.current_boxes)

    def pop_data(self):
        if self.n_buffered > 0:
            self.recent_boxes.pop()

    def set_current_boxes(self, boxes):
        self.current_boxes = boxes

    def get_all_boxes(self):
        allboxes = []
        for boxes in self.recent_boxes:
            allboxes += boxes
        if len(allboxes) == 0:
            self.allboxes = None
        else:
            self.allboxes = allboxes

    def update(self, boxes):
        self.set_current_boxes(boxes)
        self.add_boxes()
        self.get_all_boxes()


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    if bbox_list:
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap



def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img




boxes = BoundingBoxes(n=30)
def process_image(image):
    draw_image = np.copy(image)
    image = image.astype(np.float32) / 255
    hot_windows, _ = search_all_scales(image)
    boxes.update(hot_windows)
    heatmap = np.zeros_like(image[:, :, 0]).astype(np.float)
    heatmap = add_heat(heatmap, boxes.allboxes)
    heatmap = apply_threshold(heatmap, 15)
    labels = label(heatmap)

    window_image = draw_labeled_bboxes(draw_image, labels)
    return window_image


# Video Processing
print('Processing the video')

out_dir = './output_images/'
inpfile = 'project_video.mp4'
outfile = out_dir + 'processed_' + inpfile
clip = VideoFileClip(inpfile)
out_clip = clip.fl_image(process_image)
tqdm(out_clip.write_videofile(outfile, audio=False))

print('Done')

