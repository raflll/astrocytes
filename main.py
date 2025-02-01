import os
from skimage import io, filters
import numpy as np
from pathlib import Path

# Optimal threshold found to be .6-.7, further optimization can be done if desired
THRESHOLD = 0.64 # Coefficient to multiply Otsu's value (Lower value to increase features, raise to decrease noise)

def binarize_image(image_path, output_path, threshold):
    # Read the image
    image = io.imread(image_path)

    # Apply Otsu's thresholding but lower it by multiplying by a factor
    threshold = filters.threshold_otsu(image) * threshold  # Lowered to desired threshold
    binary = image > threshold

    # Convert boolean to uint8 with 0 and 255 values
    binary_image = binary.astype(np.uint8) * 255

    # Save the binarized image
    io.imsave(output_path, binary_image)

def process_directory(base_path, THRESHOLD):
    # Create output directory if it doesn't exist
    output_base = "binarized_images"
    if not os.path.exists(output_base):
        os.makedirs(output_base)

    # Process all directories
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.tiff'):
                # Get full file paths
                file_path = Path(root) / file

                # Create corresponding output directory
                rel_path = os.path.relpath(root, base_path)
                output_dir = Path(output_base) / rel_path
                output_dir.mkdir(parents=True, exist_ok=True)

                output_path = output_dir / file

                # Binarize image
                binarize_image(str(file_path), str(output_path), THRESHOLD)
                print(f"Processed: {file}")

if __name__ == "__main__":
    base_path = "data"
    process_directory(base_path, THRESHOLD)
    print("Binarization complete.")
