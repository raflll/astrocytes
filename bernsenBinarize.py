import cv2
import numpy as np
from skimage import io, filters
from scipy import ndimage

def bersen_binarize_image(image_path, output_path):
    # TODO: Make SIZE_FILTER dynamic depending on the image? I'm not sure if this is possible or necessary
    SIZE_FILTER = 100  # Lower if we are not detecting small cells, raise if we are getting noise

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Contrast Limited Adaptive Histogram Equalization to boost contrast before applying Bernsen threshold
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))  # Clip limit can be optimized (2-3 is optimal)
    enhanced = clahe.apply(img)

    # Apply Bernsen's threshold using skimage
    threshold_value = filters.threshold_local(enhanced, block_size=25, method='mean', offset=None, mode='reflect', param=None)
    binary = enhanced > threshold_value
    binary = np.uint8(binary) * 255  # Convert to uint8

    # Filter out noise
    labeled_array, num_features = ndimage.label(binary)
    component_sizes = np.bincount(labeled_array.ravel())
    too_small = component_sizes < SIZE_FILTER
    too_small_mask = too_small[labeled_array]
    binary[too_small_mask] = 0

    # Close small gaps
    # kernel = np.ones((3, 3), np.uint8)
    # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    io.imsave(output_path, binary)

    return binary