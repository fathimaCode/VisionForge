import numpy as np
import cv2
import matplotlib.pyplot as plt
# 11. **Sharpen Image using Kernels** – Enhance edges using sharpening kernels. 

def sharpen_image(image):
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return kernel

img = cv2.imread('E:/VisionForge/raw_vision/road.jpg',0)
kernel = sharpen_image(img)
h,w = img.shape
kh,kw = kernel.shape
pad_h, pad_w = kh // 2, kw // 2

padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')

#cv2.imshow('Original Image',padded)
result = np.zeros_like(img, dtype=np.float32)
for i in range(h):
    for j in range(w):
        region = padded[i:i+kh,j:j+kw]
        top_left = (j, i)
        bottom_right = (j + kw - 1, i + kh - 1)
        cv2.rectangle(padded_bgr, top_left, bottom_right, (0, 0, 255), 1)
cv2.imshow('Original Image',padded_bgr)
# plt.figure(figsize=(12, 6))

# plt.subplot(1,2,1)
# plt.title("Original Image")
# plt.imshow(img)
# plt.axis('off') 
# plt.subplot(1,2,2)
# plt.title("RGB Image")
# plt.imshow(img[:,:,::-1])
# plt.axis('off') 









cv2.waitKey(0)
cv2.destroyAllWindows()
 
# 12. **Emboss Effect** – Create an embossed effect with a custom kernel.  
# 13. **Motion Blur Effect** – Simulate motion blur using a linear kernel.  
# 14. **Binarization (Global Thresholding)** – Convert an image to black and white based on a threshold.  
# 15. **Adaptive Thresholding** – Perform pixel-based thresholding for uneven lighting.  
# 16. **Background Subtraction** – Subtract static background from an image.  
# 17. **Object Segmentation (Manual Masking)** – Segment objects using pixel masks.  
# 18. **Edge Detection with Laplacian Filter** – Highlight edges using the Laplacian operator.  
# 19. **Gradient Magnitude Visualization** – Visualize gradients from Sobel filters.  
# 20. **Region of Interest (ROI) Extraction** – Extract a specific region from an image.