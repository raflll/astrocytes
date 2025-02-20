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
import copy

def process_image(file_path, binarized_dir, skeleton_dir):
    # Make strings from the combined paths
    binarized_path = binarized_dir / file_path.name
    skeleton_path = skeleton_dir / file_path.name

    # Binarize image
    binarize_image(str(file_path), str(binarized_path))

    # Apply skeletonization
    pruned_skeleton = apply_skeletonization(str(binarized_path), str(skeleton_path))

    # Extract features
    features = extract_all_features(str(binarized_path), pruned_skeleton)

    return str(file_path.name), features

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

    # Skeletonize and then prune image
    complete_skeleton = pcv.morphology.skeletonize(mask=binary_img)
    pruned_complete_skeleton = pcv.morphology.prune(skel_img=complete_skeleton, size=PRUNE_SIZE)

    savable_pruned_skeleton = (pruned_complete_skeleton[0] > 0).astype(np.uint8) * 255

    # Save to skeletonized file and return pruned skeleton
    cv2.imwrite(skeletonized_file, savable_pruned_skeleton)
    return pruned_complete_skeleton


def extract_all_features(binarized_file, pruned_complete_skeleton):
    # load binarized image and convert to np.uint8
    img = cv2.imread(binarized_file, cv2.IMREAD_GRAYSCALE)
    binary_img = img.astype(np.uint8)

    # Run connected componenets
    num_labels, labels = cv2.connectedComponents(binary_img)

    # Make a list of all the labels
    labels_list = list(range(1, num_labels))  # Start from 1 to skip background
    results = []
    all_features = []

    # Add the features from each feature to results
    for label in labels_list:
        results.append((process_astrocyte(label, labels, pruned_complete_skeleton)))

    # Add reults to all features
    for r in results:
        if not r is None: all_features.append(r)

    print(f"Processed: {binarized_file}")

    return all_features if all_features else [{"object_label": 0, "num_branches": 0, "num_projections" : 0, "branch_lengths": [],
            "total_skeleton_length": 0, "area": 0, "fractal_dim": 0, "perimeter": 0, "circularity": 0, "roundness": 0,
            "projection_lengths": [], "num_neighbors" : 0, "length_width_ratio" : 0}]

def process_astrocyte(label, labels, pruned_complete_skeleton):
    # Extract mask of astrocyte
    astrocyte_mask = (labels == label).astype(np.uint8) * 255

    if np.sum(astrocyte_mask) == 0:
        return None # Skip empty masks

    # Skeletonize individual astrocyte
    individual_skeleton = pruned_complete_skeleton[0] & (astrocyte_mask > 0)

    # Convert to format compatible with skan
    skeleton_uint8 = (individual_skeleton > 0).astype(np.uint8) * 255

    if np.sum(skeleton_uint8) == 0:
        return None  # Skip empty skeletons

    try:
        # Feature extraction
        skeleton_obj = Skeleton(skeleton_uint8)
        skeleton_data = summarize(skeleton_obj, separator='-')

        # Calculate perimeter
        contours, _ = cv2.findContours(astrocyte_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = cv2.arcLength(contours[0], True) if len(contours) > 0 else 0

        # LENGTH/WIDTH RATIO feature
        x, y, width, height = cv2.boundingRect(astrocyte_mask)
        length = max(width, height)
        width = min(width, height)
        length_width_ratio = length / width if width > 0 else 0

        # Nikhita's FRACTAL DIMENSION
        num_projs2, proj_lengths, avg_proj_length, max_proj_length = analyze_projections(skeleton_uint8)
        if num_projs2 > 0:
            fractal_dim = calculate_fractal_dimension(astrocyte_mask) # see function
        else:
            fractal_dim = 0 # astrocyte with no projections is not fractal, so FD does not apply

        if fractal_dim > 1.6: # function returned large fractal dimension for small, low-res images, so getting rid of them
            fractal_dim = -1

        # Nikhita's NEIGHBORS feature
        kernel = np.ones((70, 70), np.uint8) # need to determine the area to consider for neighbors
        dilated_mask = cv2.dilate(astrocyte_mask, kernel, iterations=1)
        neighbors = np.unique(labels[dilated_mask > 0])
        neighbors = neighbors[(neighbors != label) & (neighbors != 0)]  # removing current component and 0 (background)
        num_neighbors = len(neighbors)


        # Collect features for this astrocyte
        features = {
            "object_label": label,
            "num_branches": len(skeleton_data) if len(skeleton_data) > 0 else 0,
            "num_projections" : num_projs2,
            "branch_lengths": skeleton_data['branch-distance'].tolist() if len(skeleton_data) > 0 else [],
            "total_skeleton_length": sum(skeleton_data['branch-distance']) if len(skeleton_data) > 0 else 0,
            "area": np.sum(astrocyte_mask > 0),
            "fractal_dim": fractal_dim,
            "perimeter": perimeter,
            "circularity": (4 * np.pi * np.sum(astrocyte_mask > 0)) / (perimeter ** 2) if perimeter > 0 else 0,
            "roundness": perimeter**2 / (4 * math.pi * np.sum(astrocyte_mask > 0)),
            "projection_lengths": proj_lengths,
            "num_neighbors" : num_neighbors,
            "length_width_ratio" : length_width_ratio
        }

        # Return both features and the pruned skeleton
        return features

    except Exception as e:
        # print(f"Warning: Error processing label {label}: {str(e)}")
        return None

def analyze_projections(skeleton_component):

    segments, objs = pcv.morphology.segment_skeleton(skel_img=skeleton_component)

    # segment sort seems pretty accurate, run debug to visualize
    # pcv.params.debug = "plot"
    projection_objs, body_objs = pcv.morphology.segment_sort(skel_img=skeleton_component,
                                                  objects=objs)
    # NUMBER OF PROJECTIONS feature
    num_projs = len(projection_objs)

    # getting ALL PROJECTION LENGTHS feature
    pcv.params.sample_label = 'astrocyte'
    labeled_path_img = pcv.morphology.segment_path_length(segmented_img=segments, objects=projection_objs) # objs must be projections only
    proj_lengths = pcv.outputs.observations['astrocyte']['segment_path_length']['value']

    # AVG/MAX PROJECTION LENGTHS FEATURE
    avg_proj_length = np.mean(proj_lengths) if proj_lengths else 0
    max_proj_length = np.max(proj_lengths) if proj_lengths else 0

    return num_projs, proj_lengths, avg_proj_length, max_proj_length

def calculate_fractal_dimension(skeleton_img, min_box=1, max_box=None):
    # Computing fractal dimension w box counting, found to be the best method
    if max_box is None: max_box = min(skeleton_img.shape) // 2 # max box size is half the image, following convention

    sizes = np.logspace(np.log10(min_box), np.log10(max_box), num=10, dtype=int)
    counts = np.array([boxcount(skeleton_img, s) for s in sizes if s > 0])
    counts = counts

    coeffs = np.polyfit(np.log(sizes[:len(counts)]), np.log(counts), 1)
    return -coeffs[0] * 1.3 # Negative slope is fractal dimension, had to inflate

def boxcount(skeleton_img, k):
    # Count non-empty boxes of size k in binary image Z (box counting method)
    S = np.add.reduceat(
        np.add.reduceat(skeleton_img, np.arange(0, skeleton_img.shape[0], k), axis=0),
        np.arange(0, skeleton_img.shape[1], k), axis=1
    )
    return np.count_nonzero(S)
