from preprocessing import *
from postprocessing import *
import os

def process_directory(base_path):
    # Create output directories if they don't exist
    binarized_base = "binarized_images"
    skeleton_base = "skeletonized_images"
    if not os.path.exists(binarized_base): os.makedirs(binarized_base)
    if not os.path.exists(skeleton_base): os.makedirs(skeleton_base)

    features = {}  # Change to dictionary with image paths as keys

    # Process all directories
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.tiff'):
                rel_path = os.path.relpath(root, base_path)
                image_key = os.path.join(rel_path, file)
                features[image_key] = process_image(root, file, base_path,
                                                 binarized_base, skeleton_base)

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
