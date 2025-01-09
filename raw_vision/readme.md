# Raw Vision: Image Processing Pipeline

This project demonstrates an interactive image processing pipeline using [Gradio](https://gradio.app/). It allows users to upload an image and view various image transformations and filtering techniques applied to it in real time.

## Features

The following image processing techniques are implemented:

1. **Grayscale Conversion**  
   Converts the uploaded RGB image into a grayscale image using luminance weights.

2. **Contrast Adjustment**  
   Enhances the contrast of the image by applying a linear transformation.

3. **Histogram Equalization**  
   Redistributes image intensity values to improve contrast and brightness.

4. **Negative Transformation**  
   Creates a negative image by inverting the pixel values.

5. **Edge Detection**  
   Detects edges in the grayscale image using the Sobel operator.

6. **Noise Addition**  
   Adds random Gaussian noise to simulate real-world conditions.

7. **Average Filtering**  
   Smoothens the noisy image using a 3x3 averaging kernel.

8. **Median Filtering**  
   Removes noise by replacing each pixel value with the median of its neighbors.

9. **Gaussian Blur**  
   Applies Gaussian smoothing to reduce image noise and detail.

## How It Works

- **Upload an Image**: Users can upload any image through the Gradio interface.
- **See Results**: The pipeline processes the image and displays the results of each transformation.

## Installation and Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/your-repository/image-processing-pipeline.git
   cd image-processing-pipeline
2. Create a virtual environment and install dependencies:
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    pip install -r requirements.txt
    Run the application:
3. Run the appliation
    python basic_process.py
    Open the application in your browser:http://127.0.0.1:7860
**Sample Output:**
https://github.com/user-attachments/assets/4551db03-4714-4394-a138-c7bb06993849




Detail Explanation on this
https://medium.com/@fathima.offical.msg/raw-vision-a-scratch-based-exploration-of-image-processing-techniques-v-b6f856c7304a
