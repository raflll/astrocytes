import os
import pandas as pd
from binarize_and_skeletonize import *
from extract_features import *
from utils import save_image, save_dataframe

DATA_DIR = "../data"
OUTPUT_BINARIZED = "../binarized_images"
OUTPUT_SKELETONIZED = "../skeletonized_images"
OUTPUT_FEATURES = "../features"

FOLDERS = ["control", "treatment1", "treatment2"]

# Ensure output directories for binarized and skeletonized images exist
for folder in FOLDERS:
    os.makedirs(os.path.join(OUTPUT_BINARIZED, folder), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_SKELETONIZED, folder), exist_ok=True)
    
# Ensure output directory for features exists
os.makedirs(OUTPUT_FEATURES, exist_ok=True)

for folder in FOLDERS:
    input_folder = os.path.join(DATA_DIR, folder)
    image_files = [f for f in os.listdir(input_folder) if f.endswith(".tiff")]

    all_features = []

    for image_name in image_files:
        input_path = os.path.join(input_folder, image_name)

        # Binarize image
        original_img, binarized_img = binarize(input_path)
        save_image(binarized_img, os.path.join(OUTPUT_BINARIZED, folder, f"{image_name}"))

        # Skeletonize image
        skeletonized_img = skeletonize_and_prune(original_img, binarized_img, prune_size=10)
        save_image(skeletonized_img, os.path.join(OUTPUT_SKELETONIZED, folder, f"{image_name}"))

        # Extract features and pass the image name
        features = extract_features(binarized_img, skeletonized_img, image_name=image_name)
        all_features.extend(features)

    # Convert to DataFrame and save
    df = pd.DataFrame(all_features)
    save_dataframe(df, os.path.join(OUTPUT_FEATURES, f"{folder}_features.csv"))

print("Processing complete.")