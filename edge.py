import cv2
import numpy as np

img1 = cv2.imread('./input_data/noise.png', 0)
img2 = cv2.imread('./input_data/dot.jpg',0)
img1 = cv2.resize(img1, (40, 40))
img2 = cv2.resize(img2, (40, 40))
cv2.imwrite('./input_data/spec_input.png', img1)

img_edge = cv2.Sobel(img1, cv2.CV_64F, 1, 1, ksize=3)
img_edge = np.uint8(np.abs(img_edge))

cv2.imwrite('./input_data/spec_gt.png', img_edge)
cv2.imshow('edge', img_edge)
cv2.waitKey(0)