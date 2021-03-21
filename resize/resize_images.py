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
Size = 128

# フォルダ作成
file_dir = glob.glob("./exreference")
new_dir = './exreference_128x128'.format(Size)
os.makedirs(new_dir, exist_ok=True)

# 画像検索
files = glob.glob(file_dir[0]+'/*.png')
print(' {0} files exit.'.format(len(files)))

# リサイズ
size = (Size,Size)
for file in files:

    img = Image.open(file)
    name = os.path.basename(file)

    img = crop_max_square(img)

    img_resize = img.resize(size, Image.LANCZOS)
    img_resize.save(new_dir+'/'+name)
