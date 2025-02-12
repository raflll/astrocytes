import cv2
from skimage.filters import unsharp_mask
import numpy as np
from scipy import ndimage
from skimage import io
from skan import Skeleton, summarize
from plantcv import plantcv as pcv
import os
from pathlib import Path

def process_image(root, file, base_path, binarized_base, skeleton_base, features):
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

def apply_skeletonization(binarized_file, skeletonized_file):
    #TODO: MAKE NOT SCUFFED
    PRUNE_SIZE = 10

    # load binarized image and convert to np.uint8
    img = cv2.imread(binarized_file, cv2.IMREAD_GRAYSCALE)
    binary_img = img.astype(np.uint8)

    # Run connected componenets
    num_labels, labels = cv2.connectedComponents(binary_img)

    # Create an empty image for all skeletons
    all_skeletons = np.zeros_like(binary_img)

    all_features = []

    # Process each astrocyte individually
    for label in range(1, num_labels):  # Start from 1 to skip background
        # Create mask for current astrocyte
        astrocyte_mask = (labels == label).astype(np.uint8) * 255

        # Skeletonize individual astrocyte
        skeleton = pcv.morphology.skeletonize(mask=astrocyte_mask)
        pruned_skeleton = pcv.morphology.prune(skel_img=skeleton, size=PRUNE_SIZE)

        # Add to complete skeleton image
        all_skeletons |= (pruned_skeleton[0] * 255).astype(np.uint8)

        # Convert to format compatible with skan
        skeleton_uint8 = (pruned_skeleton[0] > 0).astype(np.uint8) * 255

        try:
            # Check if skeleton is empty
            if np.sum(skeleton_uint8) == 0:
                print(f"Warning: Empty skeleton for label {label}")
                continue

            # Feature extraction
            skeleton_obj = Skeleton(skeleton_uint8, keep_images=True)  # Add keep_images=True
            skeleton_data = summarize(skeleton_obj, separator='-')  # Specify separator explicitly

            # Calculations for perimeter
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
                "roundness": (4 * np.pi * np.sum(astrocyte_mask > 0)) / (perimeter ** 2) if perimeter > 0 else 0
                # Roundness is really circularity but biologists want to call it "Roundness" for some reason
            }

            all_features.append(features)

        except ValueError as e:
            print(f"Warning: Skipping label {label} due to invalid skeleton structure: {str(e)}")
            continue
        except Exception as e:
            print(f"Warning: Error processing label {label}: {str(e)}")
            continue

    # Save complete skeleton image
    cv2.imwrite(skeletonized_file, all_skeletons)

    return all_features if all_features else [{"object_label": 0, "num_branches": 0, "branch_lengths": [],
                                             "total_skeleton_length": 0, "area": 0, "perimeter": 0, "roundness": 0}]
