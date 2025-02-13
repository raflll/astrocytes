import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from tifffile import imsave
from scipy.integrate import quad
import skimage as ski
from scipy.signal import find_peaks
from skimage.filters import unsharp_mask
from scipy import ndimage


def get_tifs():
    return sorted([os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith(".TIFF") or file.endswith(".tiff")])

def binarize_justin(img):
    # TODO: Make SIZE_FILTER dynamic depending on the image? I'm not sure if this is possible or necessary
    SIZE_FILTER = 75 # Lower if we are not detecting small cells, raise if we are getting noise

    # apply unsharp mask filter
    img = (unsharp_mask(img, radius=20, amount=2) * 255).astype(np.uint8)

    # Apply TRIANGLE threshold
    _, binary = cv.threshold(img, 0, 255, cv.THRESH_TRIANGLE)

    # Filter out noise
    labeled_array, num_features = ndimage.label(binary)
    component_sizes = np.bincount(labeled_array.ravel())
    too_small = component_sizes < SIZE_FILTER
    too_small_mask = too_small[labeled_array]
    binary[too_small_mask] = 0

    # Close small gaps
    # kernel = np.ones((3, 3), np.uint8)
    # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    return binary

def binarize_new(image):
    # blur = cv.GaussianBlur(image,(5,5),0)
    # _, binary = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    # image = cv.fastNlMeansDenoising(image,None,10,10,7,21)
    thresh = ski.filters.threshold_otsu(image)
    binary = image > thresh
    return binary

def analyze_image_histogram(image):
    # Calculate histogram
    hist, bin_edges = np.histogram(image.ravel(), bins=100, range=(0, 256))
    
    # # Find peaks
    peaks, _ = find_peaks(hist, height=max(hist)*0.001)  # Adjust threshold as needed
    
    # # Find leftmost and rightmost peaks
    left_peak = peaks[0]
    right_peak = peaks[-1]
    
    # # Find minimum between peaks
    valley_region = hist[left_peak:right_peak]
    valley_position = left_peak + np.argmin(valley_region)
    
    # # Create the plot
    plt.figure(figsize=(12, 6))
    
    # # Plot histogram
    plt.hist(image.ravel(), bins=100, range=(0,50), color='gray', alpha=0.7)

    # print(image.ravel())
    
    # Plot vertical lines at peaks
    plt.axvline(x=left_peak, color='red', linestyle='--', 
                label=f'Left Peak (value={left_peak})')
    plt.axvline(x=right_peak, color='blue', linestyle='--', 
                label=f'Right Peak (value={right_peak})')
    
    # # Plot valley line
    plt.axvline(x=valley_position, color='green', linestyle='--', 
                label=f'Valley (value={valley_position})')
    
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Image Histogram with Peak and Valley Detection')
    plt.legend()
    
    return

input_dir = 'test'
images = get_tifs()
images = [cv.imread(img, cv.IMREAD_GRAYSCALE) for img in images][2:4]
analyze_image_histogram(images[1])

# fig, ax = ski.filters.try_all_threshold(images[1], figsize=(10,8), verbose=False)
# plt.show()

# Create the plot
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
plt.subplots_adjust(hspace=-0.2, wspace=0.1) 

# Set column titles
axes[0, 0].set_title('Original Images')
axes[0, 1].set_title('Binarized Images Justin')
axes[0, 2].set_title('Binarized Images New')

for i, image in enumerate(images):
    # analyze_image_histogram(image)
    # Original image
    axes[i, 0].imshow(image, cmap='gray')
    axes[i, 0].axis('off')

    # Binarized image old
    bin_image_old = binarize_justin(image)
    axes[i, 1].imshow(bin_image_old, cmap='gray')
    axes[i, 1].axis('off')

    # Binarized image new
    bin_image_new = binarize_new(image)
    axes[i, 2].imshow(bin_image_new, cmap='gray')
    axes[i, 2].axis('off')

plt.show()
