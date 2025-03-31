import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage as ski
from skimage.filters import unsharp_mask
from skimage.morphology import (remove_small_objects, remove_small_holes, skeletonize)
from scipy.ndimage import label as label
from plantcv import plantcv as pcv


def get_image(input_path, unsharp=False):
    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    if unsharp:
        image = (unsharp_mask(image, radius=20, amount=2) * 255).astype(np.uint8)

    return image


# Can set binarization method, using new method right now (2.13.25)
def binarize(input_path, method="latest", plot=False):
    if method == "latest":
        orig_image = get_image(input_path)
        x, y, w, h = cv2.boundingRect(orig_image)

        # Getting nucleus image
        ch1_path = input_path.replace("ch2", "ch1")  # Assume same path structure
        image_ch1 = get_image(ch1_path)
        if image_ch1 is None:
            raise FileNotFoundError(f"Could not find corresponding nucleus image: {ch1_path}")
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

        binarized_img = (final_mask > 0).astype(np.uint8) * 255
        
    elif method == "new":
        image = get_image(input_path)

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
        
        binarized_img = (output_mask > 0).astype(np.uint8) * 255

    if method == "otsu":
        image = get_image(input_path, unsharp=True)

        otsu_thresh, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_filled = remove_small_holes(otsu.astype(bool), area_threshold=1000).astype(np.uint8) * 255
        otsu_denoised = remove_small_objects(otsu_filled.astype(bool), min_size=75).astype(np.uint8) * 255
        
        # Dynamic threshold calculation for Canny edge detection
        _, max_val, _, _ = cv2.minMaxLoc(image)  # intensity range of the image
        low_thresh = int(0.18 * max_val)  # lower threshold = 25% of max intensity
        high_thresh = int(0.35 * max_val)
        
        # print(f"Dynamic Low Canny Threshold: {low_thresh}")
        # print(f"Dynamic High Canny Threshold: {high_thresh}")

        edges = cv2.Canny(image, low_thresh, high_thresh).astype(np.uint8)
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        edges_filled = remove_small_holes(edges.astype(bool), area_threshold=100).astype(np.uint8) * 255
        edges_denoised = remove_small_objects(edges_filled.astype(bool), min_size=75).astype(np.uint8) * 255

        binarized_img = cv2.bitwise_or(otsu_denoised, edges_denoised)
    
    if method == "triangle":
        image = get_image(input_path, unsharp=True)

        _, triangle = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        triangle_filled = remove_small_holes(triangle.astype(bool), area_threshold=100).astype(np.uint8) * 255
        binarized_img = remove_small_objects(triangle_filled.astype(bool), min_size=75).astype(np.uint8) * 255

    
    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(15, 15))
        fig.suptitle(input_path, fontsize=16, ha='center', va='top', y=0.72)

        ax[0].imshow(image, cmap="gray")
        ax[0].set_title("Original  (unsharp)")
        ax[0].axis("off")

        ax[1].imshow(binarized_img, cmap="gray")
        ax[1].set_title("Otsu Binarized")
        ax[1].axis("off")

    return image, binarized_img


def skeletonize_and_prune(original_img, binarized_img, prune_size, plot=False):
    
    skeleton = skeletonize(binarized_img > 0).astype(np.uint8) * 255
    
    # pcv.params.debug = "plot"

    skeleton_pruned, _, _ = pcv.morphology.prune(skel_img=skeleton, size=prune_size) # using prune size 10 for now

    if plot:
        fig, ax = plt.subplots(1, 3, figsize=(15, 15))
        
        ax[0].imshow(original_img, cmap="gray")
        ax[0].set_title("Original (unsharp)")
        ax[0].axis("off")

        ax[1].imshow(skeleton, cmap="gray")
        ax[1].set_title("Skeleton")
        ax[1].axis("off")

        ax[2].imshow(skeleton_pruned, cmap="gray")
        ax[2].set_title("Pruned Skeleton")
        ax[2].axis("off")

        plt.show()

    return skeleton_pruned
