import glob
import time
import pickle
import cv2
import numpy as np
import matplotlib.image as mpimg
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from lesson_functions import *

def train(color_space='YCrCb', orient = 9, pix_per_cell=8, cell_per_block=2,
                    hog_channel='ALL', spatial_size = (16, 16), hist_bins=32,
                    spatial_feat=True, hist_feat=True, hog_feat=True):

# color_space='YCrCb'
# orient=9
# pix_per_cell=8
# cell_per_block=2
# hog_channel='ALL'
# spatial_size=(16, 16)
# hist_bins=32
# spatial_feat=True
# hist_feat=True
# hog_feat=True



    cars_filename = glob.glob('vehicles/**/*.png')
    not_cars_filename = glob.glob('non-vehicles/**/*.png')

    print('Started the feature extraction...')
    start = time.time()

    car_features = extract_features(cars_filename, cspace=color_space,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=spatial_feat,
                                hist_feat=hist_feat, hog_feat=hog_feat)

    notcar_features = extract_features(not_cars_filename, cspace=color_space,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=spatial_feat,
                                hist_feat=hist_feat, hog_feat=hog_feat)
    end = time.time()
    print('{:.2f} Seconds to extract features'.format(end - start))

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2,
                                                        random_state=rand_state)
    print('Using:',orient,'orientations',pix_per_cell, 'pixels per cell and',
        cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))

    svc = LinearSVC()
    t=time.time()
    svc.fit(X_train, y_train)
    t2=time.time()

    t=time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')


    dist_pickle = {}
    dist_pickle["color_space"] = color_space
    dist_pickle["svc"] = svc
    dist_pickle["scaler"] = X_scaler
    dist_pickle["orient"] = orient
    dist_pickle["pix_per_cell"] = pix_per_cell
    dist_pickle["cell_per_block"] = cell_per_block
    dist_pickle["spatial_size"] = spatial_size
    dist_pickle["hog_channel"] = hog_channel
    dist_pickle["hist_bins"] = hist_bins
    dist_pickle["spatial_feat"] = spatial_feat
    dist_pickle["hist_feat"] = hist_feat
    dist_pickle["hog_feat"] = hog_feat

    with open("svc_pickle.p", "wb") as f:
        pickle.dump(dist_pickle, f)

train()
