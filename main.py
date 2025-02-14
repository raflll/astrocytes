import copy
import os
from skimage import io, morphology, measure, img_as_ubyte
from skimage.filters import unsharp_mask
import numpy as np
from pathlib import Path
import glob
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from skan import Skeleton
from concurrent.futures import ThreadPoolExecutor
import skimage as ski
from scipy.ndimage import label

def binarize_image(image_path, output_path):

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

def process_image(file_path, binarized_dir, skeleton_dir):
    binarized_path = binarized_dir / file_path.name
    skeleton_path = skeleton_dir / file_path.name

    # Binarize image
    binarize_image(str(file_path), str(binarized_path))

    # Apply skeletonization and return features
    # In future, let's try to break this up
    features = apply_skeletonization(str(binarized_path), str(skeleton_path))
    
    return features

def process_directory(base_path, image_extensions={".tiff", ".tif", ".png"}):
    # Create output directory if it doesn't exist
    base_path = Path(base_path)
    binarized_base = Path("binarized_images")
    skeleton_base = Path("skeletonized_images")

    binarized_base.mkdir(exist_ok=True)
    skeleton_base.mkdir(exist_ok=True)

    # Collect all image paths first
    # This allows us to process images in parallel later
    image_paths = []
    for file_path in base_path.rglob("*"):
        if file_path.suffix.lower() in image_extensions:
            rel_path = file_path.parent.relative_to(base_path)
            binarized_dir = binarized_base / rel_path
            skeleton_dir = skeleton_base / rel_path

            binarized_dir.mkdir(parents=True, exist_ok=True)
            skeleton_dir.mkdir(parents=True, exist_ok=True)

            image_paths.append((file_path, binarized_dir, skeleton_dir))

    # Process images in parallel
    features = []
    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda args: process_image(*args), image_paths)
        features.extend(results)

    return features

def setup_extract_features(skeletonized, binarized, data, features):
    for subfolder in os.listdir(binarized):
        binarized_subfolder_path = os.path.join(binarized, subfolder)
        data_subfolder_path = os.path.join(data, subfolder)
        skeletonized_subfolder_path = os.path.join(skeletonized, subfolder)

        # Ensure the corresponding subfolder exists in data
        if os.path.isdir(binarized_subfolder_path) and os.path.isdir(data_subfolder_path):

            print(f"Processing subfolder: {subfolder}")
            # Get only files ending in .tiff to avoid errors
            binarized_images = glob.glob(os.path.join(binarized_subfolder_path, "*.tiff"))

            # Extract features from the binarized images
            extract_features(skeletonized_subfolder_path, binarized_images, data_subfolder_path, features)

def extract_features(skeleton_images, binarized_images, data_subfolder_path, features):
    for i, binarized_image_path in enumerate(binarized_images):
        # Construct corresponding image path in data folder
        image_filename = os.path.basename(binarized_image_path)
        data_image_path = os.path.join(data_subfolder_path, image_filename)
        skeleton_image_path = os.path.join(skeleton_images, image_filename)

        # Ensure the corresponding image exists
        if os.path.exists(data_image_path) and os.path.exists(skeleton_image_path):

            # Load images
            binarized_image = cv2.imread(binarized_image_path, cv2.IMREAD_GRAYSCALE)
            data_image = cv2.imread(data_image_path, cv2.IMREAD_GRAYSCALE)
            skeleton_image = cv2.imread(skeleton_image_path, cv2.IMREAD_GRAYSCALE)

            # Count number of little guys :3
            num_labels, labels = cv2.connectedComponents(binarized_image)

            # Format plot and image for visualization
            plt.figure(figsize=(10, 5))
            data_image = cv2.cvtColor(data_image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for visualization
            binarized_image = cv2.cvtColor(binarized_image, cv2.COLOR_GRAY2BGR) # Convert to BGR for visualization

            # Add rectangles and skeleton over original image
            blended = blend(binarized_image, skeleton_image)
            rectangle(binarized_image, labels, num_labels)

            plt.subplot(1, 2, 2)
            plt.imshow(data_image, cmap='gray')
            plt.title(f"Data: {image_filename}")
            plt.axis("off")

            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
            plt.title(f"Astrocyte Count: {num_labels - 1}")
            plt.axis("off")

            print(get_features(features, i))
            plt.show()

            # TODO: Use the extracted features from the skeletonization feature extraction

        else:
            print(f"Matching file not found in data for: {image_filename}")

def get_features(features, i):

    Totals = {
        "num_branches": 0,
        "branch_lengths" : 0,
        "total_skeleton_length" : 0,
        "most_branches" : 0
    }

    for f in features[i]:
        Totals["num_branches"] += f["num_branches"]
        Totals["branch_lengths"] += sum(f["branch_lengths"])
        Totals["total_skeleton_length"] += f["total_skeleton_length"]
        Totals["most_branches"] = max(Totals["most_branches"], f["num_branches"])


    Averages = {
        "num_branches" : Totals["num_branches"] / len(features[i]),
        "branch_lengths" : Totals["branch_lengths"] / Totals["num_branches"],
        "skeleton_length" : Totals["total_skeleton_length"] / len(features[i]),
        "most_branches" : Totals["most_branches"]
    }

    return Averages

# Can we break this function up?
# Ideally, we would have one function for skeletonization and one for feature extraction
def apply_skeletonization(binarized_file, skeletonized_file):
    img = cv2.imread(binarized_file, cv2.IMREAD_GRAYSCALE)

    # Define label mask and create skeleton
    label_mask = measure.label(img)
    skeleton = morphology.skeletonize(img) # Can switch to Lee instead of Zhang? I'm not sure if we want 2D or 3D

    skeleton_save = img_as_ubyte(skeleton)
    io.imsave(skeletonized_file, skeleton_save, check_contrast=False)

    features = []

    # Feature extraction
    for region in measure.regionprops(label_mask):
        mask = label_mask == region.label

        # Focus on one object in the skeletonized image
        skeleton_region = skeleton * mask

        # Only analyze astrocytes with more than 1 pixel in their skeleton
        if np.sum(skeleton_region) <= 1: continue
        skel = Skeleton(skeleton_region)

        # Get features
        num_branches = len(skel.paths_list())
        branch_lengths = [len(path) for path in skel.paths_list()]
        total_skeleton_length = np.sum(branch_lengths)

        # Add features of individual object
        features.append({
            "object_label": region.label,
            "num_branches": num_branches,
            "branch_lengths": branch_lengths,
            "total_skeleton_length": total_skeleton_length,
        })

    return features

def blend(original, skeleton):
    og = copy.deepcopy(original)
    skeleton_mask = skeleton > 0
    # Set original image to red where the skeleton mask exists to overlay skeleton on original
    og[skeleton_mask] = [0, 0, 255]
    return og

def rectangle(data_image, labels, num_labels):
    for label in range(1, num_labels):  # Start from 1 to ignore background
        mask = (labels == label).astype(np.uint8) * 255  # Get individual component
        x, y, w, h = cv2.boundingRect(mask)
        cv2.rectangle(data_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green bounding box


if __name__ == "__main__":
    # Binarize images
    base_path = "data"
    features = process_directory(base_path)
    print("Binarization complete.")

    # TODO: By deafult, we should simply save a .csv with all of the data
    # TODO: We should only show these individual images if the user sets debug=True

    # Extract features from the binarized images
    binarized_path = "binarized_images"
    skeleton_path = "skeletonized_images"
    setup_extract_features(skeleton_path, binarized_path, base_path, features)
