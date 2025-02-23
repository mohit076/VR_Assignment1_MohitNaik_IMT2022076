import os
import cv2
import numpy as np

# Preprocesses the image: converts to grayscale, resizes, blurs, and applies binary thresholding
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = 700 / max(image.shape[:2])
    image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    grayscale = cv2.resize(grayscale, (0, 0), fx=scale, fy=scale)
    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    binary_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return image, binary_thresh, scale

# Detects coin-like shapes in the binary thresholded image using contour analysis
def detect_coins(binary_thresh, scale):
    shapes, _ = cv2.findContours(binary_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coin_shapes = []
    for shape in shapes:
        perimeter = cv2.arcLength(shape, True)
        area = cv2.contourArea(shape)
        if perimeter:
            circularity = 4 * np.pi * (area / (perimeter ** 2))
            if 0.7 < circularity < 1.2 and area > 500 * (scale ** 2):
                coin_shapes.append(shape)
    return coin_shapes

# Segments the detected coins while keeping the original background
def segment_coins(image, coin_shapes):
    segmented = image.copy()
    cv2.drawContours(segmented, coin_shapes, -1, (0, 0, 255), 2)
    return segmented

# Extracts individual coins, crops them, and saves them with a black background
def extract_coin(image, coin, output_path, filename, index):
    x, y, w, h = cv2.boundingRect(coin)
    coin_mask = np.zeros_like(image)
    cv2.drawContours(coin_mask, [coin], -1, (255, 255, 255), thickness=cv2.FILLED)
    segmented_coin = cv2.bitwise_and(image, coin_mask)
    black_bg = np.zeros_like(image)
    black_bg[coin_mask > 0] = segmented_coin[coin_mask > 0]
    cropped_coin = black_bg[y:y+h, x:x+w]
    base_filename = os.path.splitext(filename)[0]
    cv2.imwrite(f"{output_path}/segmented/{base_filename}_{index}.jpg", cropped_coin)

# Processes all images in the input folder, applies segmentation, and extracts individual coins
def process_images(input_folder, output_folder):
    os.makedirs(f"{output_folder}/outlined", exist_ok=True)
    os.makedirs(f"{output_folder}/segmented", exist_ok=True)
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, "outlined", os.path.splitext(filename)[0] + "_outline.jpg")
        image, binary_thresh, scale = preprocess_image(input_path)
        coin_shapes = detect_coins(binary_thresh, scale)
        segmented_coins = segment_coins(image, coin_shapes)
        cv2.imwrite(output_path, segmented_coins)
        for i, coin in enumerate(coin_shapes):
            extract_coin(image, coin, output_folder, filename, i+1)
        print(f"{filename}: Total coins detected = {len(coin_shapes)}")

# Main function to execute the image processing pipeline
def main():
    process_images("input", "output")

if __name__ == "__main__":
    main()