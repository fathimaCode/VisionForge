# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 16:38:07 2025

@author: fathimaCode
"""

import numpy as np
import cv2

# RGB to Gray
def rgb2gray(rgb_image):
    return np.dot(rgb_image[...,:3],[0.299,0.587,0.114])



rgb_img = cv2.imread('road.jpg')
gray_img = rgb2gray(rgb_img)
gray_clip = np.clip(gray_img,0,255).astype(np.uint8)
#cv2.imshow('Gray Image',gray_clip)


#adjust Brightness
threshold =5
def onChange(value):
    global rgb_img
    rgb_img = np.clip(rgb_img+value, 0, 255)
    cv2.imshow('Brightness Image',rgb_img)
    print(value)

#cv2.imshow('Brightness Image',rgb_img)
#cv2.createTrackbar('slider', 'Brightness Image', 10, 250, onChange)


#contrast changes
alpha = 18
img_mean = np.mean(gray_clip)
print(img_mean)
contrast =np.clip( alpha * (gray_clip - img_mean) + img_mean,0,255)
cv2.imshow('Contrast Image',contrast)
cv2.waitKey(0)
cv2.destroyAllWindows()