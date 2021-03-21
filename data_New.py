 #!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import math
import joblib
import glob
import re


import cv2

import numpy as np
import numpy.random as rd

from settings_New import settings

import os

#コントラスト類似度計算
import contrastSimilarity as cont

#ヒストグラム平坦化
import histgramEqualize as hist



# -------------------------------------------
#   Load pkl files or ".jpg" & ".csv" files
# -------------------------------------------
def data_loader(test=False):
    """
    Read wav files or Load pkl files
	"""

    ##  Sort function for file name
    def numericalSort(value):
        numbers = re.compile(r'(\d+)')
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

	# Load settings
    args = settings()

    # Make folder
    if not os.path.exists(args.model_save_path):    # Folder of model
        os.makedirs(args.model_save_path)

    if not os.path.exists(args.pkl_path):           # Folder of train pkl
        os.makedirs(args.pkl_path)

    ex = "bmp"

    # File name
    if not test:
        image_names = args.train_data_path + '/*.' + "png" #すべてのtrainBMPファイルを読み込み
        eval_names  = args.train_data_path + '/*.txt' #すべてのtraintxtファイルを読み込み
        pkl_image   = args.pkl_path + '/train_image.pkl'
        pkl_eval    = args.pkl_path + '/train_eval.pkl'
        pkl_mask = args.pkl_path + '/train_mask.pkl'
        mask_names  = args.train_mask_path  + '/*.' + "png"
    else:
        image_names = args.test_data_path + '/*.' + "bmp"
        eval_names  = args.test_data_path + '/*.txt'
        pkl_image   = args.pkl_path + '/test_image.pkl'
        pkl_eval    = args.pkl_path + '/test_eval.pkl'
        pkl_mask = args.pkl_path + '/test_mask.pkl'
        mask_names = args.test_mask_path + '/*.' + "bmp"


    image_data = []
    mask_data = []
    similarity  = []


    ##  ~~~~~~~~~~~~~~~~~~~
    ##   No pkl files
    ##    -> Read images & assesment values, and Create pkl files
    ##  ~~~~~~~~~~~~~~~~~~~
    if not (os.access(pkl_image, os.F_OK) and os.access(pkl_eval, os.F_OK) and os.access(pkl_mask, os.F_OK)):

        ##  Read Image files
        print(' Load bmp file...')

        # Get image data

        for image_file in sorted(glob.glob(image_names), key=numericalSort):
            img = cv2.imread(image_file)
            #image_data.append(img.transpose(2,0,1))
            image_data.append(img)

        image_data = np.array(image_data)

        # Get evaluation data
        eval_data = []
        for imgage_file in sorted(glob.glob(eval_names), key=numericalSort):
            eval_data = np.expand_dims(np.loadtxt(glob.glob(eval_names)[0], delimiter=',', dtype='float'), axis=1)



        for mask_file in sorted(glob.glob(mask_names), key=numericalSort):
            mask = cv2.imread(mask_file)
                #mask_data.append(mask.transpose(2, 0, 1))
            mask_data.append(mask)

        mask_data = np.array(mask_data)



        ##  Create Pkl files
        print(' Create Pkl file...')
        with open(pkl_image, 'wb') as f:        # Create clean pkl file
            joblib.dump(image_data, f, protocol=-1, compress=3)

        with open(pkl_eval, 'wb') as f:         # Create noisy pkl file
            joblib.dump(eval_data, f, protocol=-1, compress=3)

        with open(pkl_mask, 'wb') as f:  # Create clean pkl file
            joblib.dump(mask_data, f, protocol=-1, compress=3)

    else: #pklファイルの読み込み
        #if test  == False:  #train_pkl
            with open(pkl_image, 'rb') as f:        # Load image pkl file
                print(' Load Image Pkl...')
                image_data = joblib.load(f)

            with open(pkl_eval, 'rb') as f:         # Load evaluation pkl file
                print(' Load Evaluation Pkl...')
                eval_data = joblib.load(f)

            with open(pkl_mask, 'rb') as f:  # Load image pkl file
                print(' Load Mask Pkl...')
                mask_data = joblib.load(f)

    #コントラスト類似度を取得　(画像枚数,3) & ヒストグラム平坦化
    histimg = []
    histmask = []
    for i in range (5670):
        img1 = image_data[i]
        img2 = mask_data[i]
        similarity.append(cont.contrastSimilarity(img1, img2))
        histimg.append(hist.histgramEqualize(img1).transpose(2, 0, 1))
        histmask.append(hist.histgramEqualize(img2).transpose(2, 0, 1))


    image_data = np.concatenate((histimg, histmask), axis=1)

    return image_data, eval_data , similarity


class create_batch:
    """
    Creating Batch Data for training
    """

    ## 	Initialization
    def __init__(self, image, mos, batches,image_files =[], test=False):

        # Data Shaping #.copyで値渡しを行う => 代入だと参照渡しになってしまう
        self.image  = image.copy()
        self.mos    = mos.copy()
        self.image_files = image_files.copy()


        # Random index ( for data scrambling)

        ind = np.array(range(self.image.shape[0]))
        if not test:
            rd.shuffle(ind)

        # Parameters
        self.i = 0
        self.batch = batches
        self.iter_n = math.ceil(self.image.shape[0] / batches)     # Batch num for each 1 Epoch
        remain = self.iter_n * batches - self.image.shape[0]
        self.rnd = np.r_[ind, np.random.choice(ind, remain)] # Reuse beggining of data when not enough data

    def shuffle(self):
        self.i = 0
        rd.shuffle(self.rnd)

    def __iter__(self):
        return self

    ## 	Pop batch data
    def __next__(self):

        self.test = False

        if not self.test:
            index = self.rnd[self.i  * self.batch: (self.i + 1) * self.batch ]   # Index of extracting data
            self.i += 1
            #return self.image[index], self.mos[index], self.image_files[index[0]]  # Image & MOS
            return self.image[index], self.mos[index]   # Image & MOS

        else:
            index = self.rnd[self.i * self.batch: (self.i + 1) * self.batch]  # Index of extracting data
            self.i += 1

            return self.image[index]  # Image

