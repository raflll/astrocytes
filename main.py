import os
from skimage import io, filters, measure, exposure, restoration
import numpy as np
from pathlib import Path
import glob
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

def binarize_image(image_path, output_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Contrast Limited Adaptive Histogram Equalization to boost contrast before applying Otsu
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8)) # Clip limit can be optimized (2-3 is optimal)
    enhanced = clahe.apply(img)

    # Apply Otsu's thresholding
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Filter out noise
    labeled_array, num_features = ndimage.label(binary)
    component_sizes = np.bincount(labeled_array.ravel())
    too_small = component_sizes < 50  # Can be optimized to filter out smaller or larger objects
    too_small_mask = too_small[labeled_array]
    binary[too_small_mask] = 0

    # Close small gaps
    kernel = np.ones((5,5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    io.imsave(output_path, binary)

    return binary


def process_directory(base_path):
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
                binarize_image(str(file_path), str(output_path))
                print(f"Processed: {file}")

def setup_extract_features(binarized, data, spam = True):
    # Spam turns on or off the side by side visualization of the data & it's binarized counterpart
    for subfolder in os.listdir(binarized):
        binarized_subfolder_path = os.path.join(binarized, subfolder)
        data_subfolder_path = os.path.join(data, subfolder)

        # Ensure the corresponding subfolder exists in data
        if os.path.isdir(binarized_subfolder_path) and os.path.isdir(data_subfolder_path):

            print(f"Processing subfolder: {subfolder}")
            # Get only files ending in .tiff to avoid errors
            binarized_images = glob.glob(os.path.join(binarized_subfolder_path, "*.tiff"))

            # Extract features from the binarized images
            extract_features(binarized_images, data_subfolder_path, spam)

# TODO: REFACTOR FOR READABILITY
def extract_features(binarized_images, data_subfolder_path, spam):
    for binarized_image_path in binarized_images:
        # Construct corresponding image path in data folder
        image_filename = os.path.basename(binarized_image_path)
        data_image_path = os.path.join(data_subfolder_path, image_filename)

        # Ensure the corresponding image exists
        if os.path.exists(data_image_path) and spam:
            # Load images
            binarized_image = cv2.imread(binarized_image_path, cv2.IMREAD_GRAYSCALE)
            data_image = cv2.imread(data_image_path, cv2.IMREAD_GRAYSCALE)

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

            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
            plt.title(f"Binarized\nAstrocyte Count: {num_labels - 1}")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(data_image, cmap='gray')
            plt.title(f"Data: {image_filename}")
            plt.axis("off")

            plt.show()

        else:
            print(f"Matching file not found in data for: {image_filename}")



if __name__ == "__main__":
    # Binarize images
    base_path = "data"
    process_directory(base_path)
    print("Binarization complete.")

    # Extract features from the binarized images
    extract_path = "binarized_images"
    setup_extract_features(extract_path, base_path, spam=True)
