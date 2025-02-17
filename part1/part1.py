import cv2
import numpy as np
import os

# Function to preprocess the image: convert to grayscale, resize, blur, and apply binary thresholding
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale_factor = 700 / max(image.shape[:2])
    image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
    grayscale = cv2.resize(grayscale, (0, 0), fx=scale_factor, fy=scale_factor)
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    binary_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return image, binary_thresh, scale_factor

# Function to detect coin-like shapes in the binary thresholded image using contour analysis
def detect_coins(binary_thresh, scale_factor):
    shapes, _ = cv2.findContours(binary_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coin_shapes = []
    for shape in shapes:
        perimeter = cv2.arcLength(shape, True)
        area = cv2.contourArea(shape)
        if perimeter:
            circularity = 4 * np.pi * (area / (perimeter ** 2))
            if 0.7 < circularity < 1.2 and area > 500 * (scale_factor ** 2):
                coin_shapes.append(shape)
    return coin_shapes

# Function to apply region-based segmentation on coins
def segment_coins(image, binary_thresh, coin_shapes):
    mask = np.zeros_like(binary_thresh)
    cv2.drawContours(mask, coin_shapes, -1, 255, thickness=cv2.FILLED)
    segmented = cv2.bitwise_and(image, image, mask=mask)
    black_background = np.zeros_like(image)
    black_background[mask == 255] = segmented[mask == 255]
    return black_background

# Function to process all images, apply segmentation, and draw contours
def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + "_outline.jpg")

        image, binary_thresh, scale_factor = preprocess_image(input_path)
        coin_shapes = detect_coins(binary_thresh, scale_factor)
        segmented_coins = segment_coins(image, binary_thresh, coin_shapes)

        cv2.drawContours(segmented_coins, coin_shapes, -1, (0, 0, 255), 2)
        cv2.imwrite(output_path, segmented_coins)
        print(f"{filename}: Total coins detected = {len(coin_shapes)}")

process_images("input", "output")
