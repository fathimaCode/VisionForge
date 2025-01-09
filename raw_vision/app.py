# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 16:38:07 2025

@author: fathimaCode
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal

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
img_mean = np.mean(rgb_img)
#print(img_mean)
contrast =np.clip( alpha * (rgb_img - img_mean) + img_mean,0,255)
#cv2.imshow('Contrast Image',contrast)


#Histogram Equalisation
flatten_img = rgb_img.flatten()
hist, bins =np.histogram(flatten_img,256,[0,256])
#Calculate the cumulative distribution function (CDF)
cdf = hist.cumsum()
#print(f'Cumulative distrubution fun:{cdf.shape}')
#normalise
cdf_normalized = cdf * 255 / cdf[-1]
img_eq = np.interp(rgb_img.flatten(), bins[:-1], cdf_normalized).reshape(rgb_img.shape)
img = img_eq.astype(np.uint8)


                
binary_img = np.where(gray_clip >= 215, 255, 0).astype(np.uint8)

#Negative Image Transformation

negative_image = 255 -  rgb_img

#Edge Detection
Kx = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])  
Ky = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])  

px = np.array([[-1,0,1], [-1,0,1], [-1,0,1]])
py = np.array([[-1,-1,-1], [0,0,0], [1,1,1]])


print(Kx.shape)

sobel_x = signal.convolve2d(gray_clip, Kx, mode='same')
sobel_y = signal.convolve2d(gray_clip, Ky, mode='same')

edges = np.hypot(sobel_x, sobel_y)
threshold_value = 50
binary_edges = np.where(edges >= threshold_value, 255, 0).astype(np.uint8)


#Average Filtering


kernel = np.ones((3,3))/9
print(kernel)
averge_blur =signal.convolve2d(gray_clip, kernel, mode='same')


#Guassian Blur
from scipy.ndimage import gaussian_filter,median_filter
gauss_blur = gaussian_filter(gray_clip, sigma=1)
gauss_blur1 = gaussian_filter(gray_clip, sigma=3)


#noise image
noise_img = np.random.normal(0,25,gray_clip.shape)
make_noise_img = gray_clip+noise_img


#applying average and gaussian blur
averge_blur =signal.convolve2d(make_noise_img, kernel, mode='same')
gauss_blur = gaussian_filter(make_noise_img, sigma=1)
gauss_blur1 = gaussian_filter(make_noise_img, sigma=3)


#median Filtering
denoise = median_filter(make_noise_img,size=3)

plt.figure(figsize=(10, 5))


plt.subplot(1, 4, 1) 
plt.imshow(make_noise_img, cmap='gray')
plt.title("Original Noise Image")
plt.axis('off') 

# Second subplot
plt.subplot(1, 4, 2)
plt.imshow(make_noise_img, cmap='gray')
plt.title("Average Blur")
plt.axis('off') 

# Second subplot
plt.subplot(1,4, 3)
plt.imshow(denoise, cmap='gray')
plt.title("Median Filter k=3")
plt.axis('off') 


plt.subplot(1,4, 4)
plt.imshow(gauss_blur1, cmap='gray')
plt.title("Gaussian Blur 1")
plt.axis('off') 


plt.tight_layout()  
plt.show()




cv2.waitKey(0)
cv2.destroyAllWindows()