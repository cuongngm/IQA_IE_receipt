import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('demo/1.jpg')
scan = cv2.imread('demo/scanner.jpg')
nor = cv2.imread('demo/normalize.jpg')
patch = cv2.imread('demo/patch.jpg')
titles = ['Original Image', 'scanner', 'normalize', 'patch','TOZERO']
images = [img, scan, nor, patch]
for i in range(4):
    plt.subplot(1, 4, i+1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()