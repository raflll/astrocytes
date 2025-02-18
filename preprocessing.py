import cv2
from skimage.filters import unsharp_mask
import numpy as np
from scipy import ndimage
from skimage import io
from skan import Skeleton, summarize
from plantcv import plantcv as pcv
import os
from pathlib import Path
import math
import skimage as ski
from scipy.ndimage import label
from concurrent.futures import ThreadPoolExecutor

def process_image(file_path, binarized_dir, skeleton_dir):
    binarized_path = binarized_dir / file_path.name
    skeleton_path = skeleton_dir / file_path.name

    # Binarize image
    binarize_image(str(file_path), str(binarized_path))

    # Apply skeletonization and return features
    features = apply_skeletonization(str(binarized_path), str(skeleton_path))

    return features

def binarize_image(image_path, output_path):
    SIZE_FILTER = 75 # Lower if we are not detecting small cells, raise if we are getting noise

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # apply unsharp mask filter
    img = (unsharp_mask(img, radius=20, amount=2) * 255).astype(np.uint8)

    # Apply TRIANGLE threshold
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_TRIANGLE)

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

def binarize_image_new(image_path, output_path):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    thresh = ski.filters.threshold_triangle(image)
    binary_full = image > thresh
    binary_full = (binary_full * 255).astype(np.uint8)

    # Define the kernel for dilation
    kernel = np.ones((5, 5), np.uint8)

    binary_full = cv2.morphologyEx(binary_full, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(binary_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask to store the filtered particles
    final_mask = np.zeros_like(binary_full)

    # Loop through contours and keep only large ones
    for cnt in contours:
        contour_mask = np.zeros_like(binary_full)
        cv2.drawContours(contour_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        # Dilate the contour
        dilated_mask = cv2.dilate(contour_mask, kernel, iterations=1)

        # Find bounding rectangle of the dilated mask
        x, y, w, h = cv2.boundingRect(dilated_mask)

        # Crop the original image to the bounding box
        cropped_region = image[y:y+h, x:x+w]
        cropped_dilated_mask = dilated_mask[y:y+h, x:x+w]

        enhance_contrast = 10
        cropped_region = np.clip(cropped_region.astype(np.int32) * enhance_contrast, 0, 255).astype(np.uint8)

        # Apply triangle thresholding to the masked region
        block_size = w//2
        if block_size%2 == 0:
            block_size = block_size + 1
        thresh = ski.filters.threshold_local(cropped_region, block_size=block_size)
        binary_region = cropped_region > thresh
        binary_region = (binary_region * 255).astype(np.uint8)

        # Label connected components in binary_region
        labeled_region, num_features = label(binary_region)

        # Create an empty array to store the resulting components
        kept_components = np.zeros_like(binary_region)

        # Iterate through each component and check for overlap with cropped_dilated_mask
        for i in range(1, num_features + 1):
            # Create a mask for the current component
            component_mask = (labeled_region == i)

            # Check if there is any overlap with cropped_dilated_mask
            if np.any(component_mask & cropped_dilated_mask):  # If overlap exists
                kept_components[component_mask] = 255  # Keep the component

        # Add the thresholded cropped region to the final mask at the correct location
        final_mask[y:y+h, x:x+w] = cv2.bitwise_or(final_mask[y:y+h, x:x+w], kept_components)

    # Fill holes
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Compute sizes (areas) of all detected particles
    particle_sizes = np.array([cv2.contourArea(cnt) for cnt in contours])
    size_thresh = ski.filters.threshold_triangle(particle_sizes)

    # Initialize the output mask
    output_mask = np.zeros_like(binary_full)

    # Draw contours that have an area above the threshold
    for cnt in contours:
        if cv2.contourArea(cnt) > size_thresh:  # Only draw if the area is above the threshold
            cv2.drawContours(output_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    io.imsave(output_path, output_mask)

    return output_mask

def apply_skeletonization(binarized_file, skeletonized_file):
    PRUNE_SIZE = 3

    # load binarized image and convert to np.uint8
    img = cv2.imread(binarized_file, cv2.IMREAD_GRAYSCALE)
    binary_img = img.astype(np.uint8)

    # Run connected componenets
    num_labels, labels = cv2.connectedComponents(binary_img)

    # Create an empty image for all skeletons
    all_skeletons = np.zeros_like(binary_img)

    complete_skeleton = pcv.morphology.skeletonize(mask=binary_img)
    pruned_complete_skeleton = pcv.morphology.prune(skel_img=complete_skeleton, size=PRUNE_SIZE)

    labels_list = list(range(1, num_labels))  # Start from 1 to skip background
    all_features = []
    all_individual_skeletons = []

    # Use ThreadPoolExecutor to parallelize feature extraction
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda label: process_astrocyte(label, labels, pruned_complete_skeleton), labels_list))

    # Process results: collect features and merge skeletons
    for feature, skeleton in results:
        if feature is not None:
            all_features.append(feature)
        if skeleton is not None:
            all_individual_skeletons.append(skeleton)

    # Merge all individual skeletons into all_skeletons
    for skeleton in all_individual_skeletons:
        all_skeletons |= skeleton

    # Save complete skeleton image
    cv2.imwrite(skeletonized_file, all_skeletons)

    print(f"Processed: {binarized_file}")

    return all_features if all_features else [{"object_label": 0, "num_branches": 0, "branch_lengths": [],
                                               "total_skeleton_length": 0, "area": 0, "perimeter": 0, "roundness": 0}]

def process_astrocyte(label, labels, pruned_complete_skeleton):
    # Extract mask of astrocyte
    astrocyte_mask = (labels == label).astype(np.uint8) * 255

    if np.sum(astrocyte_mask) == 0:
        return None, None  # Skip empty masks

    # Skeletonize individual astrocyte
    individual_skeleton = pruned_complete_skeleton[0] & (astrocyte_mask > 0)

    # Convert to format compatible with skan
    skeleton_uint8 = (individual_skeleton > 0).astype(np.uint8) * 255

    if np.sum(skeleton_uint8) == 0:
        return None, None  # Skip empty skeletons

    try:
        # Feature extraction
        skeleton_obj = Skeleton(skeleton_uint8)
        skeleton_data = summarize(skeleton_obj, separator='-')

        # Calculate perimeter
        contours, _ = cv2.findContours(astrocyte_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = cv2.arcLength(contours[0], True) if len(contours) > 0 else 0

        # Collect features for this astrocyte
        features = {
            "object_label": label,
            "num_branches": len(skeleton_data) if len(skeleton_data) > 0 else 0,
            "branch_lengths": skeleton_data['branch-distance'].tolist() if len(skeleton_data) > 0 else [],
            "total_skeleton_length": sum(skeleton_data['branch-distance']) if len(skeleton_data) > 0 else 0,
            "area": np.sum(astrocyte_mask > 0),
            "perimeter": perimeter,
            "circularity": (4 * np.pi * np.sum(astrocyte_mask > 0)) / (perimeter ** 2) if perimeter > 0 else 0,
            "roundness": perimeter**2 / (4 * math.pi * np.sum(astrocyte_mask > 0))
        }

        # Return both features and the pruned skeleton
        return features, (individual_skeleton * 255).astype(np.uint8)

    except Exception as e:
        print(f"Warning: Error processing label {label}: {str(e)}")
        return None, None
