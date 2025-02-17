# VR_Assignment1_MohitNaik_IMT2022076

### Part 1: Use Computer Vision techniques to detect, segment, and count coins from an image containing scattered Indian coins.

- **Preprocessing** - Converted image to grayscale, resized to a fixed scale, applied Gaussian blurring to reduce noise and used adaptive thresholding to create a binary image where coins are highlighted.

- **Region-based Segmentation** - Isolated and detected individual coins using contours and circularity measures.

- **Output** - Images with coin outlines are saved, and the total number of coins present in the images is printed on the terminal.

- **Results and Observations** - Performed well with various camera angles and scales. Performed poorly with dark backgrounds and closely spaced coins.

#### **How to Run**

Requirements - `python` with  `numpy` and `opencv` installed

In the `part1` directory, the `input` directory contains the input images labeled `0.jpg`,`1.jpg` etc. 

To run the code, use the command

```
python3 part1.py
```

This will output the number of coins in each image on the terminal, while saving the coin outlines for each image in the `output` directory, labeled `0_outline.jpg`,`1_outline.jpg` etc.

---

### Part 2: Create a stitched panorama from multiple overlapping images.

- Used **SIFT (Scale-Invariant Feature Transform)** to detect keypoints and extract descriptors.

- Used **Brute-Force Matcher (BFMatcher)** to find correspondences between keypoints in overlapping images and **Lowe's ratio test** to filter out poor matches.

- Computed the **homography** using **RANSAC (Random Sample Consensus)** to warp one image onto another.

- Stitched the images together followed by **blending** to remove seams, **cropping** to remove black regions and **resizing** for consistency.

- **Output** - A single **panorama** image created by stitching the overlapping input images, and a set of images showing the **matching key points** between any two images.

- **Results and Observations** - Performs well in most of the cases, fails to give a good panorama when input images are not properly aligned or have very small overlap.

#### **How to Run**

Requirements - `python` with  `numpy` and `opencv` installed

In the `part2` directory, the `input1` and `input2` directorys contain two distinct sets of overlapping images, labeled `0.jpg`,`1.jpg` etc. The input directory to be used can be selected by changing the last line of `part2.py`.

To run the code, use the command

```
python3 part2.py
```

This will save the panorama image as `panorama.jpg` in the `output` directory. It will also save the matching keypoint images as `match_1.jpg`,`match_2.jpg` etc.

---

Github: https://github.com/mohit076/VR_Assignment1_MohitNaik_IMT2022076
