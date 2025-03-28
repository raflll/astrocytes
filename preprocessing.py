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

def process_directory(input_path, image_extensions={".tiff", ".tif", ".png"}):
    # Convert input path to Path object
    input_path = Path(input_path)

    # Check if input path exists
    if not input_path.exists():
        print(f"Error: Path '{input_path}' does not exist")
        return None

    # Create output directories if they don't exist
    binarized_base = Path("binarized_images")
    skeleton_base = Path("skeletonized_images")

    # Also create directory for extracted features
    features_dir = Path("extracted_features")

    binarized_base.mkdir(exist_ok=True)
    skeleton_base.mkdir(exist_ok=True)
    features_dir.mkdir(exist_ok=True)

    # Collect all image paths
    image_paths = []

    # Check if there are any subdirectories
    subdirs = [x for x in input_path.iterdir() if x.is_dir()]

    if subdirs:
        # Process files in subdirectories
        print(f"Found {len(subdirs)} subdirectories. Processing files in subdirectories...")
        for subdir in subdirs:
            subdir_paths = []
            for file_path in subdir.rglob("*"):
                # Skip files with "-ch1" in the filename
                if "-ch1" in file_path.name:
                    print(f"Skipping ch1 file: {file_path.name}")
                    continue

                if file_path.suffix.lower() in image_extensions:
                    rel_path = file_path.parent.relative_to(input_path)
                    binarized_dir = binarized_base / rel_path
                    skeleton_dir = skeleton_base / rel_path

                    binarized_dir.mkdir(parents=True, exist_ok=True)
                    skeleton_dir.mkdir(parents=True, exist_ok=True)

                    subdir_paths.append((file_path, binarized_dir, skeleton_dir))
            if subdir_paths:  # Only add if there are actually images
                image_paths.append(subdir_paths)
    else:
        # Process files in main directory only
        print("No subdirectories found. Processing files in main directory...")
        main_dir_paths = []
        for file_path in input_path.glob("*"):
            # Skip files with "-ch1" in the filename
            if "-ch1" in file_path.name:
                print(f"Skipping ch1 file: {file_path.name}")
                continue

            if file_path.suffix.lower() in image_extensions:
                binarized_dir = binarized_base
                skeleton_dir = skeleton_base

                main_dir_paths.append((file_path, binarized_dir, skeleton_dir))
        if main_dir_paths:  # Only add if there are actually images
            image_paths.append(main_dir_paths)

    if not image_paths:
        print(f"No images found with extensions {image_extensions}")
        return None

    num_folders = len(image_paths)
    total_images = sum(len(folder) for folder in image_paths)
    print(f"Found {total_images} images in {num_folders} folders to process")

    all_features = []
    # Process each folder
    for folder_idx, folder_paths in enumerate(image_paths):
        print(f"Processing folder {folder_idx+1}/{num_folders} with {len(folder_paths)} images")
        folder_features = {}

        # Process images in parallel within each folder
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda args: process_image(*args), folder_paths))

        # Collect results for this folder
        for result in results:
            file_name, features = result
            folder_features[file_name] = features

        all_features.append(folder_features)

    # Save features to CSV files
    for f, folder in enumerate(all_features):
        if f == 0:
            name = "extracted_features/Control_features.csv"
        elif f == 1:
            name = "extracted_features/Images_features.csv"
        elif f == 2:
            name = "extracted_features/Phenotype 1_features.csv"
        elif f == 3:
            name = "extracted_features/Phenotype 2_features.csv"
        else:
            name = f"extracted_features/Folder_{f+1}_features.csv"

        # Convert nested dictionaries to DataFrame
        rows = []
        for file_name, cell_features_list in folder.items():
            for i, cell_feature in enumerate(cell_features_list):
                cell_feature['file_name'] = file_name
                rows.append(cell_feature)

        if rows:  # Only save if we have data
            df = pd.DataFrame(rows)
            save_dataframe(df, name)
            print(f"Saved features to {name}")
        else:
            print(f"No features to save for folder {f}")

def process_image(file_path, binarized_dir, skeleton_dir):
    try:
        # Make strings from the combined paths
        binarized_path = binarized_dir / file_path.name
        skeleton_path = skeleton_dir / file_path.name

        # Check if input file exists and is readable
        if not file_path.exists():
            print(f"Error: Input file {file_path} does not exist")
            return str(file_path.name), []

        # Check if we have write permissions for output directories
        if not os.access(str(binarized_dir), os.W_OK):
            print(f"Error: No write permission for binarized directory {binarized_dir}")
            return str(file_path.name), []
        if not os.access(str(skeleton_dir), os.W_OK):
            print(f"Error: No write permission for skeleton directory {skeleton_dir}")
            return str(file_path.name), []

        # Binarize image with error handling
        try:
            binarize_image(str(file_path), str(binarized_path), method="3")
        except Exception as e:
            print(f"Error binarizing image {file_path}: {str(e)}")
            return str(file_path.name), []

        # Apply skeletonization with error handling
        try:
            pruned_skeleton = apply_skeletonization(str(binarized_path), str(skeleton_path))
        except Exception as e:
            print(f"Error skeletonizing image {file_path}: {str(e)}")
            return str(file_path.name), []

        # Extract features with error handling
        try:
            features = extract_all_features(str(binarized_path), pruned_skeleton)
        except Exception as e:
            print(f"Error extracting features from {file_path}: {str(e)}")
            return str(file_path.name), []

        return str(file_path.name), features
    except Exception as e:
        print(f"Error processing image {file_path}: {str(e)}")
        return str(file_path.name), []

def binarize_image(image_path, output_path, method = "1"):
    try:
        # Read image with error checking
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        # Ensure image is in correct format
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # Process image based on method
        if method == "old": 
            print(f"Binarizing image {image_path} with method old")
            binary = binarize_old(image)
        elif method == "1": 
            print(f"Binarizing image {image_path} with method 1")
            binary = binarize_new(image)
        elif method == "2": 
            print(f"Binarizing image {image_path} with method 2")
            binary = binarize_new2(image)
        elif method == "3": 
            print(f"Binarizing image {image_path} with method 3")
            binary = binarize_new3(image)
        else:
            raise ValueError(f"Unknown binarization method: {method}")

        # Ensure output is in correct format
        if binary.dtype != np.uint8:
            binary = binary.astype(np.uint8)

        # Save with error checking
        try:
            success = io.imsave(output_path, binary)
        except Exception as e:
            print(f"Error saving image to: {output_path}")
#        if not success:
#            raise ValueError(f"Failed to save image to: {output_path}")

        return binary
    except Exception as e:
        print(f"Error in binarize_image: {str(e)}")
        raise

def binarize_old(image):
    try:
        SIZE_FILTER = 75  # Lower if we are not detecting small cells, raise if we are getting noise

        # Ensure image is in correct format
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # apply unsharp mask filter with error handling
        try:
            img = (unsharp_mask(image, radius=20, amount=2) * 255).astype(np.uint8)
        except Exception as e:
            print(f"Error in unsharp mask: {str(e)}")
            img = image.copy()  # Fallback to original image

        # Apply TRIANGLE threshold
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_TRIANGLE)

        # Filter out noise with error handling
        try:
            labeled_array, num_features = ndimage.label(binary)
            component_sizes = np.bincount(labeled_array.ravel())
            too_small = component_sizes < SIZE_FILTER
            too_small_mask = too_small[labeled_array]
            binary[too_small_mask] = 0
        except Exception as e:
            print(f"Error in noise filtering: {str(e)}")
            # Continue with unfiltered binary image

        return binary
    except Exception as e:
        print(f"Error in binarize_old: {str(e)}")
        raise

def binarize_new(image):
    try:
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
            enhance_contrast = 3
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
    except Exception as e:
        print(f"Error in binarize_new AHH: {str(e)}")
        return None 

def binarize_new2(image):
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
    kernel = np.ones((2, 2), np.uint8)
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
        radius = max(1, int(min(w, h) * 0.02)) # using small % bcs smaller radius captures more detail
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
        kernel = np.ones((2, 2), np.uint8)
        binary_region = cv2.morphologyEx(binary_region, cv2.MORPH_OPEN, kernel)
        # print(iterations)
        binary_region = cv2.morphologyEx(binary_region, cv2.MORPH_CLOSE, kernel, iterations=3)
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

def binarize_new3(image):
    x, y, w, h = cv2.boundingRect(image)

    # Getting unsharp image - dynamic radius/amount parameters
    radius = max(1, int(min(w, h) * 0.12)) # determines blur size, using higher % to preserve structure for entire img
    contrast = np.std(image)  # measuring contrast as std
    amount = np.clip((contrast / 50), 1, 3)  # normalize contrast to medium range - do not want over-sharpening for entire img yet
    image = (unsharp_mask(image, radius=radius, amount=amount) * 255).astype(np.uint8)

    full_thresh = ski.filters.threshold_triangle(image)
    # print(thresh)
    # enhance = 3 if full_thresh < 5 else 2.5
    # iterations = 3 if full_thresh < 5 else 2
    # contrast = 4 if full_thresh < 5 else 5
    binary_full = image > full_thresh
    binary_full = (binary_full * 255).astype(np.uint8)

    # print(full_thresh)

    # Dilate
    kernel = np.ones((2, 2), np.uint8)
    binary_full = cv2.morphologyEx(binary_full, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(binary_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask to store the filtered particles
    final_mask = np.zeros_like(binary_full)

    # Loop through contours and keep only large ones
    for cnt in contours:
        contour_mask = np.zeros_like(binary_full)
        cv2.drawContours(contour_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        # dilated_mask = contour_mask

        # Dilate the contour
        dilated_mask = cv2.dilate(contour_mask, kernel, iterations=1)

        # Find bounding rectangle of the dilated mask and crop
        x, y, w, h = cv2.boundingRect(dilated_mask)

        cropped_region = image[y:y+h, x:x+w]
        cropped_dilated_mask = dilated_mask[y:y+h, x:x+w]

        # # use low value
        # enhance_contrast = 2.5
        # cropped_region = np.clip(cropped_region.astype(np.int32) * enhance_contrast, 0, 255).astype(np.uint8)

        # Use unsharp to enhance specific region (instead of enhance), dynamically adjust radius
        radius = max(1, int(min(w, h) * 0.03)) # using small % bcs smaller radius captures more detail
        # print(radius)
        # Found that instead of calculating contrast dynamically, use high contrast at this stage since focusing on one component
        # and want to preserve as much detail
        # contrast = np.std(cropped_region)
        # print(contrast)
        # amount = np.clip((contrast / 5), 3, 6)
        # print(thresh)
        region_thresh = ski.filters.threshold_triangle(cropped_region)
        cropped_region = (unsharp_mask(cropped_region, radius=radius, amount=5) * 255).astype(np.uint8)

        # Apply local thresholding to the masked region
        block_size = w//2
        if block_size%2 == 0:
            block_size = block_size + 1
        thresh = ski.filters.threshold_local(cropped_region, block_size=block_size)
        binary_region = cropped_region > thresh
        binary_region = (binary_region * 255).astype(np.uint8)

        # Open to reduce noise, close to fill in gaps (helpful bcs setting contrast high preserves detail but leads to gaps)
        kernel = np.ones((2, 2), np.uint8)
        binary_region = cv2.morphologyEx(binary_region, cv2.MORPH_OPEN, kernel)
        # print(iterations)
        binary_region = cv2.morphologyEx(binary_region, cv2.MORPH_CLOSE, kernel, iterations=3)
        # binary_region = cv2.bilateralFilter(binary_region, d=3, sigmaColor=80, sigmaSpace=25)
        # binary_region = cv2.dilate(binary_region, kernel, iterations=1)

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
            "avg_projection_length": sum(proj_lengths) / len(proj_lengths),
            "max_projection_length": max(proj_lengths),
            "neighbors" : num_neighbors,
            "length_width_ratio" : length_width_ratio,
            # "mask" : astrocyte_mask
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

def save_dataframe(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
