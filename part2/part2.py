import numpy as np
import cv2
import os

# Function to detect keypoints and compute descriptors using SIFT
def detect_keypoints(image):
    descriptor = cv2.SIFT_create()
    kps, features = descriptor.detectAndCompute(image, None)
    kps = np.float32([kp.pt for kp in kps])
    return (kps, features)

# Function to match keypoints between two images using the ratio test
def match_keypoints(kpA, kpB, xA, xB, ratio, re_proj):
    matcher = cv2.BFMatcher()
    rawMatches = matcher.knnMatch(xA, xB, 2)
    matches = []
    for m in rawMatches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))
    if len(matches) > 4:
        ptsA = np.float32([kpA[i] for (_, i) in matches])
        ptsB = np.float32([kpB[i] for (i, _) in matches])  
        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, re_proj)  # Homography using RANSAC
        return (matches, H, status)
    return None

# Function to visualize the matches between two images
def draw_matches(imgA, imgB, kpA, kpB, matches, status):
    hA, wA = imgA.shape[:2]
    hB, wB = imgB.shape[:2]
    viz = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    viz[0:hA, 0:wA] = imgA
    viz[0:hB, wA:] = imgB
    line_number = 0
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        if s == 1:
            ptA = (int(kpA[queryIdx][0]), int(kpA[queryIdx][1]))
            ptB = (int(kpB[trainIdx][0]) + wA, int(kpB[trainIdx][1]))
            if line_number < 100: # Drawing only 100 lines to reduce clutter
                cv2.line(viz, ptA, ptB, (0, 0, 255), 1)
                line_number += 1
    return viz

# Function to crop the image to the region of interest (ROI) after stitching
def crop(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        return image[y:y + h - 1, x:x + w - 1]
    return image

# Function to resize an image while maintaining the aspect ratio
def resize(image, width):
    h, w = image.shape[:2]
    aspect_ratio = width / float(w)
    new_height = int(h * aspect_ratio)
    return cv2.resize(image, (width, new_height), interpolation=cv2.INTER_AREA)

# Function to stitch two images together
def stitch(images, ratio=0.75, re_proj=4.0, show_overlay=False):
    imgB, imgA = images
    kpA, xA = detect_keypoints(imgA)
    kpB, xB = detect_keypoints(imgB)
    M = match_keypoints(kpA, kpB, xA, xB, ratio, re_proj)
    if M is None:
        print("Not enough matches found.")
        return None
    matches, H, status = M
    pano_img = cv2.warpPerspective(imgA, H, (imgA.shape[1] + imgB.shape[1], imgA.shape[0]))
    pano_img[0:imgB.shape[0], 0:imgB.shape[1]] = imgB
    pano_img = crop(pano_img)
    if show_overlay:
        visualization = draw_matches(imgA, imgB, kpA, kpB, matches, status)
        return (pano_img, visualization)
    
    return pano_img

# Function to process images in a folder, stitch them together, and save the result
def process_images(input_folder, output_folder):
    img_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder)]
    img_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    left = cv2.imread(img_paths[0])
    left = resize(left, width=600)
    for i in range(1, len(img_paths)):
        right = cv2.imread(img_paths[i])
        right = resize(right, width=600)
        result = stitch([left, right], show_overlay=True)
        if result is None:
            continue
        left, viz = result
        cv2.imwrite(os.path.join(output_folder, f"{input_folder}_match_{i}.jpg"), viz)

    cv2.imwrite(os.path.join(output_folder, f"{input_folder}_panorama.jpg"), left)
    print(f"Panorama for {input_folder} created and saved in output folder.")

process_images("input1", "output")
process_images("input2", "output")