import copy
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import glob

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
        "most_branches" : 0,
        "analyzed" : 0,
        "perimeter" : 0,
        "roundness" : 0
    }

    for f in features[i]:
        Totals["num_branches"] += f["num_branches"]
        Totals["branch_lengths"] += sum(f["branch_lengths"])
        Totals["total_skeleton_length"] += f["total_skeleton_length"]
        Totals["most_branches"] = max(Totals["most_branches"], f["num_branches"])
        Totals["analyzed"] += 1
        Totals["perimeter"] += f["perimeter"]
        Totals["roundness"] += f["roundness"]


    Averages = {
        "num_branches" : Totals["num_branches"],
        "branch_lengths" : Totals["branch_lengths"] / Totals["num_branches"],
        "skeleton_length" : Totals["total_skeleton_length"] / len(features[i]),
        "most_branches" : Totals["most_branches"],
        "analyzed" : Totals["analyzed"]
    }

    return Averages
