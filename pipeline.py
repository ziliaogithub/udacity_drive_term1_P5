import numpy as np
import cv2
import time
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from lesson_functions import *
from heat import *

# 1) Perform a HOG feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
# 2) Optionally, apply a color transform and append binned color features, as well as histograms of color, to HOG feature vector.
# 3) Note: don't forget to normalize your features and randomize a selection for training and testing.
# 4) Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
# 5) Run your pipeline on a video stream (start with the test_video.mp4 and later
# implement on full project_video.mp4) and create a heat map of recurring detections
# frame by frame to reject outliers and follow detected vehicles.
# 6) Estimate a bounding box for vehicles detected.

def save_frames_from_video(file_name, number=10):
    clip = cv2.VideoCapture(file_name)
    # frames = int(clip.fps * clip.duration)
    # print(frames)
    success,image = clip.read()
    count = 0
    success = True
    while success:
        success,image = clip.read()
        print('Read a new frame: ', success)
        cv2.imwrite("test_video_frames/frame%d.jpg" % count, image)     # save frame as JPEG file
        count += 1
        if count == number:
            return

def image_with_windows(img):
    y_start_stops = [[400, 645], [400, 600], [400, 550]]
    xy_windows = [(128, 128), (96, 96), (64, 64)]
    xy_overlap=(0.5, 0.5)
    x_start_stop=[None, None]
    windows = []
    print(zip(xy_windows, y_start_stops))
    for y_start_stop, xy_window  in zip(y_start_stops, xy_windows):
        windows.extend(slide_window(img, x_start_stop=x_start_stop,
        y_start_stop=y_start_stop, xy_window=xy_window, xy_overlap=xy_overlap))

    img_with_boxes = draw_boxes(img, windows, color=(125, 255, 125), thick=4 )
    return img_with_boxes

def draw_images_with_boxes():
    img = mpimg.imread("test_images/test2.jpg")
    img = image_with_windows(img)
    plt.imshow(img)
    # plt.show()
    plt.savefig("sliding_window_examples.jpg")

def process_image(img):
    dist_pickle = pickle.load(open("svc_pickle.p", "rb" ))
    color_space = dist_pickle["color_space"]
    svc = dist_pickle["svc"]
    X_scaler = dist_pickle["scaler"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    spatial_size = dist_pickle["spatial_size"]
    hog_channel = dist_pickle["hog_channel"]
    hist_bins = dist_pickle["hist_bins"]
    spatial_feat = dist_pickle["spatial_feat"]
    hist_feat = dist_pickle["hist_feat"]
    hog_feat = dist_pickle["hog_feat"]
    y_start = 400
    y_stop = 656
    scale = 1.5

    heat = np.zeros_like(img[:, :, 0]).astype(np.float)

    hot_windows = find_cars(img, X_scaler, svc, y_start, y_stop, color_space, scale, orient,
                            pix_per_cell, hog_feat, hog_channel, cell_per_block,
                            spatial_feat, spatial_size, hist_bins, hist_feat)

    window_img = draw_boxes(img, hot_windows, color=(0, 0, 255), thick=6)

    heat = add_heat(heat, hot_windows)

    heat = apply_threshold(heat, 2)

    heatmap = np.clip(heat, 0, 255)

    labels = label(heatmap)
    draw_image = draw_labeled_bboxes(np.copy(img), labels)

    return draw_image

img = mpimg.imread("test_video_frames/frame1.jpg")
img = process_image(img)
plt.imshow(img)
plt.show()
plt.savefig("window_with_cars_examples.jpg")
input = VideoFileClip("test_video.mp4")
output = input.fl_image(process_image)
output.write_videofile('result_test.mp4', audio=False)
