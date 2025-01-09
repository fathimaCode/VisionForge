import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter, median_filter
import gradio as gr

def rgb2gray(rgb_image):
    return np.dot(rgb_image[...,:3], [0.299, 0.587, 0.114])

def process_image(image):
    # Convert RGB to grayscale
    gray_img = rgb2gray(image)
    gray_clip = np.clip(gray_img, 0, 255).astype(np.uint8)

    # Contrast adjustment
    alpha = 18
    img_mean = np.mean(image)
    contrast_img = np.clip(alpha * (image - img_mean) + img_mean, 0, 255).astype(np.uint8)

    # Histogram Equalization
    flatten_img = image.flatten()
    hist, bins = np.histogram(flatten_img, 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]
    img_eq = np.interp(image.flatten(), bins[:-1], cdf_normalized).reshape(image.shape)
    hist_eq_img = np.clip(img_eq, 0, 255).astype(np.uint8)

    # Negative Image Transformation
    negative_image = np.clip(255 - image, 0, 255).astype(np.uint8)

    # Edge Detection
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_x = signal.convolve2d(gray_clip, Kx, mode='same')
    sobel_y = signal.convolve2d(gray_clip, Ky, mode='same')
    edges = np.hypot(sobel_x, sobel_y)
    binary_edges = np.clip(np.where(edges >= 50, 255, 0), 0, 255).astype(np.uint8)

    # Noise Image
    noise_img = np.random.normal(0, 25, gray_clip.shape)
    make_noise_img = np.clip(gray_clip + noise_img, 0, 255).astype(np.uint8)

    # Filtering
    kernel = np.ones((3, 3)) / 9
    average_blur = np.clip(signal.convolve2d(make_noise_img, kernel, mode='same'), 0, 255).astype(np.uint8)
    gauss_blur = np.clip(gaussian_filter(make_noise_img, sigma=1), 0, 255).astype(np.uint8)
    gauss_blur1 = np.clip(gaussian_filter(make_noise_img, sigma=3), 0, 255).astype(np.uint8)
    denoise = np.clip(median_filter(make_noise_img, size=3), 0, 255).astype(np.uint8)

    # Return all processed images
    return (
        gray_clip,
        contrast_img,
        hist_eq_img,
        negative_image,
        binary_edges,
        make_noise_img,
        average_blur,
        denoise,
        gauss_blur1
    )

def gradio_interface(image):
   
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = process_image(image)

  
    return [
        ("Grayscale Image", results[0]),
        ("Contrast Adjusted", results[1]),
        ("Histogram Equalized", results[2]),
        ("Negative Image", results[3]),
        ("Edge Detected", results[4]),
        ("Noise Image", results[5]),
        ("Average Blur", results[6]),
        ("Median Filter", results[7]),
        ("Gaussian Blur", results[8])
    ]

def process_and_display(image):
    processed_images = gradio_interface(image)

    return [img for _, img in processed_images]

iface = gr.Interface(
    fn=process_and_display,
    inputs=gr.Image(type="numpy", label="Upload an Image"),
    outputs=[gr.Image(label=title) for title in [
        "Grayscale Image",
        "Contrast Adjusted",
        "Histogram Equalized",
        "Negative Image",
        "Edge Detected",
        "Noise Image",
        "Average Blur",
        "Median Filter",
        "Gaussian Blur"
    ]],
    title="Image Processing Pipeline",
    description="Upload an image to see various transformations and filters applied."
)

iface.launch()
