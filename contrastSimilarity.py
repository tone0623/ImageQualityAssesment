import cv2
import numpy as np



img1 = cv2.imread('./resize/eximages_128x128/I01_06_01.png')
img2 = cv2.imread('./resize/exreference_128x128/I01-1.png')

# img1 = cv2.imread('./resize/Girl/128x128/1.bmp')
# img2 = cv2.imread('./resize/Girl/128x128/1.bmp')

def  contrastSimilarity(img1, img2):

    Similarity = []

    for i in range (3):
        target_hist = cv2.calcHist([img1], [i], None, [256], [0, 256])
        comparing_hist = cv2.calcHist([img2], [i], None, [256], [0, 256])
        v = cv2.compareHist(target_hist, comparing_hist, 0)
        Similarity.append(v)

    return Similarity

# print(Similarity)
# print('Compare : {0}'.format(v))