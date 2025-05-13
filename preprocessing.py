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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
from skimage.morphology import (remove_small_objects, remove_small_holes, skeletonize)
import shutil

def cleanup_old_files():
    """
    Clean up old files from previous runs by removing contents of output directories
    and old CSV files.
    """
    # Directories to clean
    dirs_to_clean = [
        "binarized_images",
        "skeletonized_images",
        "extracted_features",
        "whole_image_features",
        "charts"
    ]
    
    # Clean each directory
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            print(f"Cleaning {dir_path}...")
            # Remove all contents but keep the directory
            for item in os.listdir(dir_path):
                item_path = os.path.join(dir_path, item)
                try:
                    if os.path.isfile(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                except Exception as e:
                    print(f"Error cleaning {item_path}: {e}")

def process_directory(input_path, image_extensions={".tiff", ".tif", ".png"}, progress_callback=None):
    # Clean up old files before starting
    cleanup_old_files()
    
    # Convert input path to Path object
    input_path = Path(input_path)

    # Check if input path exists
    if not input_path.exists():
        print(f"Error: Path '{input_path}' does not exist")
        return None

    # Create output directories if they don't exist
    binarized_base = Path("binarized_images")
    skeleton_base = Path("skeletonized_images")
    features_dir = Path("extracted_features")

    binarized_base.mkdir(exist_ok=True)
    skeleton_base.mkdir(exist_ok=True)
    features_dir.mkdir(exist_ok=True)

    # Collect all image paths
    image_paths = []

    # Check if there are any subdirectories
    subdirs = [x for x in input_path.iterdir() if x.is_dir()]

    dapi_path = None

    if subdirs:
        # Process files in subdirectories
        print(f"Found {len(subdirs)} subdirectories. Processing files in subdirectories...")
        for subdir in subdirs:
            subdir_paths = []
            for file_path in subdir.rglob("*"):
                # Skip files with "-ch1" in the filename
                if "-ch1" in file_path.name:
                    dapi_path = file_path
                    continue

                if file_path.suffix.lower() in image_extensions:
                    rel_path = file_path.parent.relative_to(input_path)
                    binarized_dir = binarized_base / rel_path
                    skeleton_dir = skeleton_base / rel_path

                    binarized_dir.mkdir(parents=True, exist_ok=True)
                    skeleton_dir.mkdir(parents=True, exist_ok=True)

                    subdir_paths.append((file_path, dapi_path, binarized_dir, skeleton_dir))
            if subdir_paths:  # Only add if there are actually images
                image_paths.append((subdir.name, subdir_paths))
    else:
        # Process files in main directory only
        print("No subdirectories found. Processing files in main directory...")
        main_dir_paths = []
        for file_path in input_path.glob("*"):
            # Skip files with "-ch1" in the filename
            if "-ch1" in file_path.name:
                dapi_path = file_path
                continue

            if file_path.suffix.lower() in image_extensions:
                binarized_dir = binarized_base
                skeleton_dir = skeleton_base

                main_dir_paths.append((file_path, dapi_path, binarized_dir, skeleton_dir))
        if main_dir_paths:  # Only add if there are actually images
            image_paths.append((input_path.name, main_dir_paths))

    if not image_paths:
        print(f"No images found with extensions {image_extensions}")
        return None

    num_folders = len(image_paths)
    total_images = sum(len(folder[1]) for folder in image_paths)
    print(f"Found {total_images} images in {num_folders} folders to process")

    all_features = []
    processed_images = 0
    # Process each folder
    for folder_idx, (folder_name, folder_paths) in enumerate(image_paths):
        print(f"Processing folder {folder_idx+1}/{num_folders} ({folder_name}) with {len(folder_paths)} images")
        folder_features = {}

        # Process images in parallel within each folder
        with ThreadPoolExecutor() as executor:
            futures = []
            for file_path, dapi_path, binarized_dir, skeleton_dir in folder_paths:
                future = executor.submit(process_image, file_path, dapi_path, binarized_dir, skeleton_dir)
                future.add_done_callback(lambda f: update_progress(f, processed_images, total_images, progress_callback))
                futures.append(future)
            
            # Wait for all futures to complete
            for future in futures:
                result = future.result()
                file_name, features = result
                folder_features[file_name] = features
                processed_images += 1

        all_features.append((folder_name, folder_features))

    # Save features to CSV files
    for folder_name, folder_features in all_features:
        # Create CSV filename based on folder name
        csv_name = f"extracted_features/{folder_name}_features.csv"

        # Convert nested dictionaries to DataFrame
        rows = []
        for file_name, cell_features_list in folder_features.items():
            for i, cell_feature in enumerate(cell_features_list):
                cell_feature['file_name'] = file_name
                rows.append(cell_feature)

        if rows:  # Only save if we have data
            df = pd.DataFrame(rows)
            save_dataframe(df, csv_name)
            print(f"Saved features to {csv_name}")
        else:
            print(f"No features to save for folder {folder_name}")

def update_progress(future, processed_images, total_images, progress_callback):
    """Update progress when an image is processed"""
    try:
        future.result()  # Get the result (this will raise any exceptions)
        if progress_callback:
            progress = int((processed_images / total_images) * 85)  # Binarization is 85% of total progress
            progress_callback(progress)
    except Exception as e:
        print(f"Error processing image: {str(e)}")

def process_image(file_path, dapi_path, binarized_dir, skeleton_dir):
    # Make strings from the combined paths
    binarized_path = binarized_dir / file_path.name
    skeleton_path = skeleton_dir / file_path.name

    # Binarize image
    binarize_image(str(file_path), str(dapi_path), str(binarized_path))

    # Apply skeletonization
    pruned_skeleton = apply_skeletonization(str(binarized_path), str(skeleton_path))

    # Extract features
    features = extract_all_features(str(binarized_path), pruned_skeleton)

    return str(file_path.name), features

def bitwise_binarize(image_path, dapi_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    dapi_image = cv2.imread(dapi_path, cv2.IMREAD_GRAYSCALE)

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = np.zeros_like(image)

    for cnt in contours:
        contour_mask = np.zeros_like(image)
        cv2.drawContours(contour_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        x, y, w, h = cv2.boundingRect(contour_mask)
        cropped_region = dapi_image[y:y+h, x:x+w]
        cropped_mask = contour_mask[y:y+h, x:x+w]
        labeled_region, num_features = label(cropped_region)
        kept_components = np.zeros_like(cropped_region)
        for i in range(1, num_features + 1):
            # Create a mask for the current component
            component_mask = (labeled_region == i)
            # Check if there is any overlap with dapi contour
            if np.any(component_mask & cropped_mask):  # If overlap exists
                kept_components[component_mask] = 255  # Keep the component
        output[y:y+h, x:x+w] = cv2.bitwise_or(output[y:y+h, x:x+w], kept_components)

    io.imsave(output_path, output)
    return output


def binarize_image(image_path, dapi_path, output_path, method = "2"):
    dapi_path = image_path.replace("ch2", "ch1")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_ch1 = cv2.imread(dapi_path, cv2.IMREAD_GRAYSCALE)

    if method == "old": binary = binarize_old(image)
    if method == "1": binary = binarize_new(image)
    if method == "2": binary = binarize_latest(image, image_ch1)

    io.imsave(output_path, binary)

    return binary

def binarize_old(image):
    SIZE_FILTER = 75 # Lower if we are not detecting small cells, raise if we are getting noise

    # apply unsharp mask filter
    img = (unsharp_mask(image, radius=20, amount=2) * 255).astype(np.uint8)

    # Apply TRIANGLE threshold
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_TRIANGLE)

    # Filter out noise
    labeled_array, num_features = ndimage.label(binary)
    component_sizes = np.bincount(labeled_array.ravel())
    too_small = component_sizes < SIZE_FILTER
    too_small_mask = too_small[labeled_array]
    binary[too_small_mask] = 0

    return binary

def binarize_new(image):
    x, y, w, h = cv2.boundingRect(image)

    # # Getting unsharp image - dynamic radius/amount parameters
    # radius = max(1, int(min(w, h) * 0.12)) # determines blur size, using higher % to preserve structure for entire img
    # contrast = np.std(image)  # measuring contrast as std
    # amount = np.clip((contrast / 50), 1, 3)  # normalize contrast to medium range - do not want over-sharpening for entire img yet
    # image = (unsharp_mask(image, radius=radius, amount=amount) * 255).astype(np.uint8)

    full_thresh = ski.filters.threshold_triangle(image)
    # print(thresh)
    # enhance = 3 if full_thresh < 5 else 2.5
    # iterations = 3 if full_thresh < 5 else 2
    # contrast = 4 if full_thresh < 5 else 5
    binary_full = image > full_thresh
    binary_full = (binary_full * 255).astype(np.uint8)

    # print(full_thresh)

    # Dilate
    kernel1 = np.ones((7, 7), np.uint8)
    binary_full = cv2.dilate(binary_full, kernel1, iterations = 1)
    kernel = np.ones((3, 3), np.uint8)
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

        # Find bounding rectangle of the dilated mask and crop
        x, y, w, h = cv2.boundingRect(dilated_mask)
        cropped_region = image[y:y+h, x:x+w]
        cropped_dilated_mask = dilated_mask[y:y+h, x:x+w]

        # use low value
        enhance_contrast = 2.5
        cropped_region = np.clip(cropped_region.astype(np.int32) * enhance_contrast, 0, 255).astype(np.uint8)

        # Use unsharp to enhance specific region (instead of enhance), dynamically adjust radius
        radius = max(1, int(min(w, h) * 0.025)) # using small % bcs smaller radius captures more detail
        # print(radius)
        # Found that instead of calculating contrast dynamically, use high contrast at this stage since focusing on one component
        # and want to preserve as much detail
        # contrast = np.std(cropped_region)
        # print(contrast)
        # amount = np.clip((contrast / 5), 3, 6)
        # print(thresh)
        region_thresh = ski.filters.threshold_triangle(cropped_region)
        cropped_region = (unsharp_mask(cropped_region, radius=radius, amount=region_thresh) * 255).astype(np.uint8)

        # Apply local thresholding to the masked region
        block_size = w//2
        if block_size%2 == 0:
            block_size = block_size + 1
        thresh = ski.filters.threshold_local(cropped_region, block_size=block_size)
        binary_region = cropped_region > thresh
        binary_region = (binary_region * 255).astype(np.uint8)

        # Open to reduce noise, close to fill in gaps (helpful bcs setting contrast high preserves detail but leads to gaps)
        kernel = np.ones((3, 3), np.uint8)
        # print(iterations)
        binary_region = cv2.morphologyEx(binary_region, cv2.MORPH_CLOSE, kernel)
        binary_region = cv2.morphologyEx(binary_region, cv2.MORPH_OPEN, kernel)
        # binary_region = cv2.bilateralFilter(binary_region, d=3, sigmaColor=80, sigmaSpace=25)


        # No change to Ethan's binarize from this point
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
    # print(particle_sizes)
    size_thresh = ski.filters.threshold_triangle(particle_sizes)

    # Initialize the output mask
    output_mask = np.zeros_like(binary_full)

    # Draw contours that have an area above the threshold
    for cnt in contours:
        if cv2.contourArea(cnt) > size_thresh:  # Only draw if the area is above the threshold
            cv2.drawContours(output_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # output_mask = remove_small_objects(output_mask.astype(bool), min_size=np.mean(particle_sizes) * 0.5).astype(np.uint8) * 255

    binarized_img = (output_mask > 0).astype(np.uint8) * 255
    return binarized_img

def binarize_latest(orig_image, image_ch1):
    x, y, w, h = cv2.boundingRect(orig_image)
    ch1_thresh, _ = cv2.threshold(image_ch1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

    # initial image sharpening
    image = (unsharp_mask(orig_image, radius=100, amount=2) * 255).astype(np.uint8)

    # binary image (global thresholding)
    full_thresh = ski.filters.threshold_triangle(image)
    binary_full = image > full_thresh
    binary_full = (binary_full * 255).astype(np.uint8)

    # Dilate to clean noise
    kernel = np.ones((2, 2), np.uint8)
    binary_full = cv2.morphologyEx(binary_full, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(binary_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Rest of your function remains the same
    # Create a mask to store the filtered particles
    final_mask = np.zeros_like(binary_full)

    # Loop through contours and keep only large ones
    for cnt in contours:
        contour_mask = np.zeros_like(binary_full)
        cv2.drawContours(contour_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        # Dilate the contour
        dilated_mask = cv2.dilate(contour_mask, kernel, iterations=1)
        # Find bounding rectangle of the dilated mask and crop
        x, y, w, h = cv2.boundingRect(dilated_mask)
        cropped_region = image[y:y+h, x:x+w]
        cropped_dilated_mask = dilated_mask[y:y+h, x:x+w]

        # Find low threshold (from nucleus image) to calculate both component and nucleus area
        triangle_thresh, _ = cv2.threshold(image_ch1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        # Component area
        _, component_bin = cv2.threshold(orig_image[y:y+h, x:x+w], triangle_thresh, 255, cv2.THRESH_BINARY)
        component_contours, _ = cv2.findContours(component_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        component_area = sum(cv2.contourArea(c) for c in component_contours)
        # Nucleus area
        _, nucleus_bin = cv2.threshold(image_ch1[y:y+h, x:x+w], triangle_thresh, 255, cv2.THRESH_BINARY)
        nucleus_contours, _ = cv2.findContours(nucleus_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        nucleus_area = sum(cv2.contourArea(c) for c in nucleus_contours)

        # Keep component only if its area is larger than the nucleus area
        if component_area <= nucleus_area and (component_area != 0):
            # print(f"Component area: {component_area}, Nucleus area: {nucleus_area}")
            continue

        # use low value to enhance image
        enhance_contrast = 1.5
        cropped_region = np.clip(cropped_region.astype(np.int32) * enhance_contrast, 0, 255).astype(np.uint8)

        # Apply local thresholding to the masked region, smaller block size preserves more detail
        block_size = w//3
        if block_size%2 == 0:
            block_size = block_size + 1
        thresh = ski.filters.threshold_local(cropped_region, block_size=block_size)
        binary_region = cropped_region > thresh
        binary_region = (binary_region * 255).astype(np.uint8)

        # Label connected components in binary_region
        labeled_region, num_features = label(binary_region)
        # Create an empty array to store the resulting components
        kept_components = np.zeros_like(binary_region)

        # Iterate through each component, check for overlap with cropped_dilated_mask AND nucleus
        for i in range(1, num_features + 1):
            # Create a mask for the current component
            component_mask = (labeled_region == i)
            # Overlap with cropped_dilated_mask (component before binarize) and nucleus (corresponding region in image_ch1)
            if (np.any(component_mask & cropped_dilated_mask) and np.any(component_mask & (image_ch1[y:y+h, x:x+w] > ch1_thresh))):
                kept_components[component_mask] = 255  # Keep the component

        # Add the thresholded cropped region to the final mask at the correct location
        final_mask[y:y+h, x:x+w] = cv2.bitwise_or(final_mask[y:y+h, x:x+w], kept_components)

    contours, hierarchy = cv2.findContours(final_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for i, cnt in enumerate(contours):
        # Check if it is a hole (hierarchy[i][3] != -1 means it's inside another contour)
        if hierarchy[0][i][3] != -1:  # It's a hole
            contour_area = cv2.contourArea(cnt)  # Calculate the area of the contour

            # Only fill if the hole is smaller than the specified threshold
            if contour_area < 75:
                cv2.drawContours(final_mask, [cnt], -1, 255, thickness=cv2.FILLED)  # Fill the hole

    binarized_img = (final_mask > 0).astype(np.uint8) * 255

    # Return the intelligently dilated image instead
    return binarized_img

def apply_skeletonization(binarized_file, skeletonized_file):
    PRUNE_SIZE = 10

    # load binarized image and convert to np.uint8
    img = cv2.imread(binarized_file, cv2.IMREAD_GRAYSCALE)
    binary_img = img.astype(np.uint8)

    # Skeletonize and then prune image
    skeleton = skeletonize(binary_img > 0).astype(np.uint8) * 255
    skeleton_pruned, _, _ = pcv.morphology.prune(skel_img=skeleton, size=PRUNE_SIZE)


    # Save to skeletonized file and return pruned skeleton
    cv2.imwrite(skeletonized_file, skeleton_pruned)
    return skeleton_pruned

def extract_all_features(binarized_file, pruned_complete_skeleton):
    # load binarized image and convert to np.uint8
    img = cv2.imread(binarized_file, cv2.IMREAD_GRAYSCALE)
    binarized_img = np.where(img > 0, 255, 0).astype(np.uint8)
    skeleton_pruned = np.where(pruned_complete_skeleton > 0, 255, 0).astype(np.uint8)

    # Run connected componenets
    num_labels, labels = cv2.connectedComponents(binarized_img)

    # Make a list of all the labels
    labels_list = list(range(1, num_labels))  # Start from 1 to skip background
    results = []
    all_features = []

    # Add the features from each feature to results
    for label in labels_list:
        results.append((process_astrocyte(label, labels, skeleton_pruned, binarized_img)))

    # Add reults to all features
    for r in results:
        if not r is None: all_features.append(r)

    print(f"Processed: {binarized_file}")

    return all_features if all_features else [{"object_label": 0, "num_branches": 0, "num_projections" : 0, "branch_lengths": [],
            "total_skeleton_length": 0, "area": 0, "fractal_dim": 0, "perimeter": 0, "circularity": 0, "roundness": 0,
            "projection_lengths": [], "num_neighbors" : 0, "length_width_ratio" : 0}]

def process_astrocyte(label, labels, pruned_complete_skeleton, binarized_img):
    # Extract mask of astrocyte
    astrocyte_mask = (labels == label).astype(np.uint8) * 255

    try:
        area = np.sum(binarized_img[astrocyte_mask > 0]) // 255

        # Feature extraction
        component_skeleton = cv2.bitwise_and(pruned_complete_skeleton, pruned_complete_skeleton, mask=astrocyte_mask)

        # Calculate perimeter
        contours, _ = cv2.findContours(astrocyte_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = sum(cv2.arcLength(cnt, True) for cnt in contours)

        # method 1 to find number of projections, I find method 2 below to be more accurate
        branch_pts = pcv.morphology.find_branch_pts(skel_img=component_skeleton)
        num_projections = np.sum(branch_pts > 0) # number of projections

        # Nikhita's FRACTAL DIMENSION - Add error handling for empty projections
        try:
            num_projs2, proj_lengths, avg_proj_length, max_proj_length = analyze_projections(component_skeleton)
            # Handle case where proj_lengths might be empty
            max_proj_length = max(proj_lengths) if proj_lengths else 0
        except Exception as e:
            print(f"Warning: Error in analyze_projections for label {label}: {str(e)}")
            num_projs2, proj_lengths, avg_proj_length, max_proj_length = 0, [], 0, 0

        # NEIGHBORS feature
        neighbors = set()
        kernel = np.ones((70, 70), np.uint8) # need to determine the area to consider for neighbors
        dilated_mask = cv2.dilate(astrocyte_mask, kernel, iterations=1)
        neighbors = np.unique(labels[dilated_mask > 0])
        neighbors = neighbors[(neighbors != label) & (neighbors != 0)]  # removing current component and 0 (background)
        num_neighbors = len(neighbors)

        # LENGTH/WIDTH RATIO feature
        x, y, width, height = cv2.boundingRect(astrocyte_mask)
        length = max(width, height)
        width = min(width, height)
        length_width_ratio = length / width if width > 0 else 0

        # FRACTAL DIMENSION feature
        skel_x, skel_y, skel_width, skel_height = cv2.boundingRect(component_skeleton)
        cropped_mask = component_skeleton[skel_y: skel_y + skel_height, skel_x: skel_x + skel_width]

        try:
            if num_projs2 > 0:
                fractal_dim = calculate_fractal_dimension(cropped_mask) # see function
            else:
                fractal_dim = 0 # astrocyte with no projections is not fractal, so FD does not apply

            if fractal_dim > 1.6: # function returned large fractal dimension for small, low-res images, so getting rid of them
                fractal_dim = -1
        except Exception as e:
            print(f"Warning: Error calculating fractal dimension for label {label}: {str(e)}")
            fractal_dim = 0

        component_skeleton = (component_skeleton > 0).astype(np.uint8)
        try:
            if np.sum(component_skeleton) == 0:
                skeleton_data = None
            else:
                try:
                    skeleton_data = summarize(Skeleton(component_skeleton), separator='-')
                except ValueError:
                    skeleton_data = None
        except Exception as e:
            print(f"Warning: Error creating skeleton data for label {label}: {str(e)}")
            skeleton_data = None

        # Safety checks for skeleton_data
        total_skeleton_length = 0

        if skeleton_data is not None:
            try:
                total_skeleton_length = sum(skeleton_data['branch-distance']) if (hasattr(skeleton_data, '__len__') and len(skeleton_data) > 0 and 'branch-distance' in skeleton_data) else 0
            except Exception as e:
                print(f"Warning: Error calculating total_skeleton_length for label {label}: {str(e)}")

        # Collect features for this astrocyte with robust error checking
        features = {
            "object_label": label,
            "num_projections": num_projs2,
            "total_skeleton_length": total_skeleton_length,
            "area": area,
            "fractal_dim": fractal_dim,
            "perimeter": perimeter,
            "circularity": (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0,
            "projection_lengths": proj_lengths if isinstance(proj_lengths, list) else [],
            "avg_projection_length": sum(proj_lengths) / len(proj_lengths) if isinstance(proj_lengths, list) and len(proj_lengths) > 0 else 0,
            "max_projection_length": max_proj_length,
            "neighbors": num_neighbors,
            "length_width_ratio": length_width_ratio
        }

        # Return both features and the pruned skeleton
        return features

    except Exception as e:
        print(f"Warning: Error processing label {label}: {str(e)}")
        return None

def analyze_projections(skeleton_component):
    try:
        segments, objs = pcv.morphology.segment_skeleton(skel_img=skeleton_component)

        # segment sort seems pretty accurate, run debug to visualize
        # pcv.params.debug = "plot"
        projection_objs, body_objs = pcv.morphology.segment_sort(skel_img=skeleton_component,
                                                      objects=objs)
        # NUMBER OF PROJECTIONS feature
        num_projs = len(projection_objs) if projection_objs is not None else 0

        # getting ALL PROJECTION LENGTHS feature
        pcv.params.sample_label = 'astrocyte'
        try:
            labeled_path_img = pcv.morphology.segment_path_length(segmented_img=segments, objects=projection_objs) # objs must be projections only
            proj_lengths = pcv.outputs.observations['astrocyte']['segment_path_length']['value']
            if proj_lengths is None:
                proj_lengths = []
        except Exception as e:
            print(f"Warning: Error getting projection lengths: {str(e)}")
            proj_lengths = []

        # AVG/MAX PROJECTION LENGTHS FEATURE
        avg_proj_length = np.mean(proj_lengths) if proj_lengths and len(proj_lengths) > 0 else 0
        max_proj_length = np.max(proj_lengths) if proj_lengths and len(proj_lengths) > 0 else 0

        return num_projs, proj_lengths, avg_proj_length, max_proj_length
    except Exception as e:
        print(f"Warning: Error in analyze_projections: {str(e)}")
        return 0, [], 0, 0

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

def save_dataframe(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
