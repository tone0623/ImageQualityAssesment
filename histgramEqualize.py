import cv2
import numpy as np
import matplotlib.pyplot as plt

# 画像の読み出し
def histgramEqualize(img):
    # BGR->HSV変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # V値のヒストグラム表示
    # hist = cv2.calcHist([hsv],[2],None,[256],[0,256])
    # plt.plot(hist,color = "b")
    # plt.show()

    # V値のヒストグラム平坦化
    hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])

    # 平坦化後のV値のヒストグラム表示
    # hist = cv2.calcHist([hsv],[2],None,[256],[0,256])
    # plt.plot(hist,color = "g")
    # plt.show()

    # HSV->BGR変換
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr
