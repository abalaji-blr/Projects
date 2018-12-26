import cv2
import numpy as np

import matplotlib.pyplot as plt

#%matplotlib inline

# opencv reads the image in BGR format
bgr_img = cv2.imread('./Images/capsicum.jpg')

gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

# hue-saturation-
hsv_img  = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

#
lab_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)

#rgb color
rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

# display them

fig, axes = plt.subplots(2,2, figsize=(10,6))
axes[0,0].imshow(rgb_img)
axes[0,0].title.set_text('RGB')

axes[0,1].imshow(gray_img)
axes[0,1].title.set_text('Gray')

axes[1,0].imshow(hsv_img)
axes[1,0].title.set_text('HSV')

axes[1,1].imshow(lab_img)
axes[1,1].title.set_text('LAB')

plt.tight_layout()
plt.show()

cv2.imshow('Gray Image', gray_img)
cv2.waitKey(0)
