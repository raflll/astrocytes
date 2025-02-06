import os
from skimage import io, morphology, measure, img_as_ubyte
import numpy as np
from pathlib import Path
import glob
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from skan import Skeleton

def binarize_image(image_path, output_path):
    SIZE_FILTER = 100 # Lower if we are not detecting small cells, raise if we are getting noise

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Contrast Limited Adaptive Histogram Equalization to boost contrast before applying Otsu
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8)) # Clip limit can be optimized (2-3 is optimal)
    enhanced = clahe.apply(img)

    # Apply TRIANGLE threshold
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_TRIANGLE)

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

def process_directory(base_path):
    # Create output directory if it doesn't exist
    binarized_base = "binarized_images"
    skeleton_base = "skeletonized_images"
    if not os.path.exists(binarized_base):
        os.makedirs(binarized_base)
    if not os.path.exists(skeleton_base):
        os.makedirs(skeleton_base)

    features = []

    # Process all directories
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.tiff'):
                # Get full file paths
                file_path = Path(root) / file

                # Create corresponding output directory
                rel_path = os.path.relpath(root, base_path)
                binarized_dir = Path(binarized_base) / rel_path
                binarized_dir.mkdir(parents=True, exist_ok=True)
                skeleton_dir = Path(skeleton_base) / rel_path
                skeleton_dir.mkdir(parents=True, exist_ok=True)

                binarized_path = binarized_dir / file
                skeleton_path = skeleton_dir / file

                # Binarize image
                binarize_image(str(file_path), str(binarized_path))
                features.append(apply_skeletonization(str(binarized_path), str(skeleton_path)))
                print(f"Processed: {file}")

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

# TODO: Use the extracted features from the skeletonization feature extraction
def extract_features(skeleton_images, binarized_images, data_subfolder_path, features):
    for binarized_image_path in binarized_images:
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

            # Plot them
            plt.figure(figsize=(10, 5))

            # Chatgpt evil visualization hack
            output_image = cv2.cvtColor(binarized_image, cv2.COLOR_GRAY2BGR)  # Convert to BGR for visualization
            for label in range(1, num_labels):  # Start from 1 to ignore background
                mask = (labels == label).astype(np.uint8) * 255  # Get individual component
                x, y, w, h = cv2.boundingRect(mask)
                cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green bounding box

            plt.subplot(1, 3, 1)
            plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
            plt.title(f"Binarized\nAstrocyte Count: {num_labels - 1}")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(skeleton_image, cmap='gray')
            plt.title("Skeleton")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(data_image, cmap='gray')
            plt.title(f"Data: {image_filename}")
            plt.axis("off")

            plt.show()

        else:
            print(f"Matching file not found in data for: {image_filename}")

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


if __name__ == "__main__":
    # Binarize images
    base_path = "data"
    features = process_directory(base_path)
    print("Binarization complete.")

    # Extract features from the binarized images
    binarized_path = "binarized_images"
    skeleton_path = "skeletonized_images"
    setup_extract_features(skeleton_path, binarized_path, base_path, features)
