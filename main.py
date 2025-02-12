from preprocessing import *
from postprocessing import *
import os

def process_directory(base_path):
    # Create output directory if it doesn't exist
    binarized_base = "binarized_images"
    skeleton_base = "skeletonized_images"
    if not os.path.exists(binarized_base): os.makedirs(binarized_base) # Create folder if it doesn't exist
    if not os.path.exists(skeleton_base): os.makedirs(skeleton_base) # Create folder if it doesn't exist

    features = []

    # Process all directories
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.tiff'):
                features.append(
                    process_image(root, file, base_path, binarized_base,
                                  skeleton_base, features))

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
