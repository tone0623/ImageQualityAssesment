# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 17:22:07 2020

@author: SUGIURA
"""

## Requirement
#   - Pillow


import os
import numpy as np
from PIL import Image
import glob

# 長方形イメージから最大サイズの正方形イメージをトリミング
def crop_max_square(pil_img):
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size)) #最小辺を入力

# イメージの中央のみをトリミング
def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

# 画像サイズ (一辺)

def compression(img):

    Size = 128

    # リサイズ
    size = (Size,Size)
    img = crop_max_square(img)
    img_resize = img.resize(size, Image.LANCZOS)

    return img_resize
