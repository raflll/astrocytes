import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.filters import (unsharp_mask, threshold_otsu, 
                             threshold_triangle, threshold_li)
from skimage.morphology import (remove_small_objects, binary_dilation, binary_erosion,
                                remove_small_holes)
from skimage.measure import perimeter, label, regionprops
import pandas as pd

##------------------------------------------------------------------------------------------------
input_base = "../data"
folder = "treatment1"  # Change here to test diff folders
input_folder = os.path.join(input_base, folder)
image_files = [f for f in os.listdir(input_folder) if f.endswith(".tiff")]

image_name = image_files[1] # Change here to test diff images
input_path = os.path.join(input_folder, image_name)

original = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
# Found lower amount value (2) to provide sufficient enhancement
unsharp_img = (unsharp_mask(original, radius=20, amount=2) * 255).astype(np.uint8)

##---- OTSU ---------------------------------------------------------------------------------------
# Also tested with "original" instead of unsharp_img, which produced worse results
# cv2 otsu produced same results as skimage otsu (same threshhold was chosen)
otsu_thresh, otsu = cv2.threshold(unsharp_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# print(f"OpenCV Otsu Threshold: {otsu_thresh}")

# Fill small holes in otsu and denoise
otsu_filled = remove_small_holes(otsu.astype(bool), area_threshold=1000).astype(np.uint8) * 255
otsu_denoised = remove_small_objects(otsu_filled.astype(bool), min_size=75).astype(np.uint8) * 255

# Compute low and high Canny thresholds
low_thresh = int(otsu_thresh * 0.5)
high_thresh = int(otsu_thresh * 1.5)
print(f"OpenCV Otsu Threshold: {otsu_thresh}")
print(f"Low Canny Threshold: {low_thresh}")
print(f"High Canny Threshold: {high_thresh}")

# Apply Canny edge detection
edges = cv2.Canny(unsharp_img, low_thresh, high_thresh).astype(np.uint8)

# Use a kernel to dilate edges and close small gaps in edges
kernel = np.ones((2, 2), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=2) # found 2 iterations was sufficient, increase for greater thickness
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Use skimage to fill in small holes in edges and denoise
edges_filled = remove_small_holes(edges.astype(bool), area_threshold=1000).astype(np.uint8) * 255
edges_denoised = remove_small_objects(edges_filled.astype(bool), min_size=75).astype(np.uint8) * 255

# Combine denoised otsu and edges
combined = cv2.bitwise_or(otsu_denoised, edges_denoised)

##---- TRIANGLE ---------------------------------------------------------------------------------------
_, triangle = cv2.threshold(unsharp_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

# Use skimage to fill in small holes in triangle and denoise
triangle_filled = remove_small_holes(triangle.astype(bool), area_threshold=700).astype(np.uint8) * 255
triangle_denoised = remove_small_objects(triangle_filled.astype(bool), min_size=75).astype(np.uint8) * 255

##------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(15, 15))
fig.suptitle(input_path, fontsize=16, ha='center', va='top', y = 0.72)

ax[0].imshow(combined , cmap="gray")
ax[0].set_title("Final Otsu")
ax[0].axis("off")

ax[1].imshow(triangle_denoised , cmap="gray")
ax[1].set_title("Final Triangle")
ax[1].axis("off")


# TODO: Use Triangle for dimmer images (otsu_threshold < 50), and otsu + edge detection otherwise?
# Repo with pipeline to binarize: https://github.com/gabys2006/TilingGlia/blob/main/Tutorial.ipynb
