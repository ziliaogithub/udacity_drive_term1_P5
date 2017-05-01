import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import glob
from lesson_functions import *
from search_and_classify import *

def create_2_columns_subplot(fst_col_imgs, snd_col_imgs):
    i = 0
    plt.figure(figsize = (10,15 ))
    for i in range(1, len(fst_col_imgs)):
        plt.subplot(6, 2, (i*2)-1)
        plt.imshow(fst_col_imgs[i])
        plt.subplot(6, 2, i*2)
        plt.imshow(snd_col_imgs[i])
    # plt.show()
    plt.savefig("output_images/result.png")

dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]

ystart = 400
ystop = 656
scale = 1.5

test_img_names = glob.glob('test_images/*.jpg')
count = len(test_img_names)
fst_col_imgs = []
snd_col_imgs = []
for test_img_name in test_img_names:
    img = mpimg.imread(test_img_name)
    fst_col_imgs.append(img)
    out_img = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    snd_col_imgs.append(out_img)
    plt.imshow(out_img)
    plt.savefig("output_images/"+test_img_name.replace("test_images/",""))
create_2_columns_subplot(fst_col_imgs, snd_col_imgs)
