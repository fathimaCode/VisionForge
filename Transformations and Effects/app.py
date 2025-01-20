import cv2
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

# Load the image in grayscale
img = cv2.imread('assets/6.jpg', 0)

# Sharpening kernel
sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

# Laplacian kernels
laplacian_kernel = np.array([[0, 1, 0],
                              [1, -4, 1],
                              [0, 1, 0]])
strong_laplacian = np.array([[-1, -1, -1],
                             [-1, 8, -1],
                             [-1, -1, -1]])
high_boost = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])

# Emboss kernel
emboss_kernel = np.array([[-2, -1, 0],
                          [-1,  1, 1],
                          [ 0,  1, 2]])

# Apply filters
sharpen_image = convolve2d(img, sharpen_kernel, mode='same', boundary='symm')
image1 = np.clip(sharpen_image, 0, 255).astype(np.uint8)

laplacian_image = convolve2d(img, laplacian_kernel, mode='same', boundary='symm')
image2 = np.clip(laplacian_image, 0, 255).astype(np.uint8)

strong_laplacian_image = convolve2d(img, strong_laplacian, mode='same', boundary='symm')
image3 = np.clip(strong_laplacian_image, 0, 255).astype(np.uint8)

high_boost_image = convolve2d(img, high_boost, mode='same', boundary='symm')
image4 = np.clip(high_boost_image, 0, 255).astype(np.uint8)

# Emboss effect
emboss_image = cv2.filter2D(img, -1, emboss_kernel)
e_normalized = cv2.normalize(emboss_image, None, 0, 255, cv2.NORM_MINMAX)

# Plot the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 5, 1)
plt.imshow(image1, cmap='gray')
plt.title('Sharpening')
plt.axis('off')

plt.subplot(1, 5, 2)
plt.imshow(image2, cmap='gray')
plt.title('Laplacian')
plt.axis('off')

plt.subplot(1, 5, 3)
plt.imshow(image3, cmap='gray')
plt.title('Strong Laplacian')
plt.axis('off')

plt.subplot(1, 5, 4)
plt.imshow(image4, cmap='gray')
plt.title('High Boost')
plt.axis('off')

plt.subplot(1, 5, 5)
plt.imshow(e_normalized, cmap='gray')
plt.title('Emboss Effect')
plt.axis('off')

plt.tight_layout()
plt.show()
