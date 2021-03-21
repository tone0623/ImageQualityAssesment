#!/usr/bin/env python
# -*- coding: utf-8 -*-
#   Image
import cv2
import numpy as np
import random


def make_input(img):
    Input = []
    Original = img
    height, width, channnel = Original.shape[:3]

    for i in range (5):
        x = random.randint(0, width - 256)
        y = random.randint(0, height - 256)

        # img[top : bottom, left : right]　256x256に切り抜き
        x2 = x + 256
        y2 = y + 256
        smallimg = Original[y : y2 , x : x2]

        Input.append(smallimg)

    Input = (np.array(Input)).transpose(0, 3, 1, 2)

    return Input
