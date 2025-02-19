from preprocessing import *
from postprocessing import *
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

def process_directory(input_path, image_extensions={".tiff", ".tif", ".png"}):
    # Convert input path to Path object
    input_path = Path(input_path)

    # Check if input path exists
    if not input_path.exists():
        print(f"Error: Path '{input_path}' does not exist")
        return None

    # Create output directories if they don't exist
    binarized_base = Path("binarized_images")
    skeleton_base = Path("skeletonized_images")

    binarized_base.mkdir(exist_ok=True)
    skeleton_base.mkdir(exist_ok=True)

    # Collect all image paths
    image_paths = []

    # Check if there are any subdirectories
    subdirs = [x for x in input_path.iterdir() if x.is_dir()]

    if subdirs:
        # Process files in subdirectories
        print(f"Found {len(subdirs)} subdirectories. Processing files in subdirectories...")
        for subdir in subdirs:
            for file_path in subdir.rglob("*"):
                if file_path.suffix.lower() in image_extensions:
                    rel_path = file_path.parent.relative_to(input_path)
                    binarized_dir = binarized_base / rel_path
                    skeleton_dir = skeleton_base / rel_path

                    binarized_dir.mkdir(parents=True, exist_ok=True)
                    skeleton_dir.mkdir(parents=True, exist_ok=True)

                    image_paths.append((file_path, binarized_dir, skeleton_dir))
    else:
        # Process files in main directory only
        print("No subdirectories found. Processing files in main directory...")
        for file_path in input_path.glob("*"):
            if file_path.suffix.lower() in image_extensions:
                binarized_dir = binarized_base
                skeleton_dir = skeleton_base

                image_paths.append((file_path, binarized_dir, skeleton_dir))

    if not image_paths:
        print(f"No images found with extensions {image_extensions}")
        return None

    print(f"Found {len(image_paths)} images to process")

    # Process images in parallel
    features = {}
    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda args: process_image(*args), image_paths)

    for file_path, r in results:
        features[file_path] = r

    return features


if __name__ == "__main__":
    # Get input path from user input
    input_path = input("Please enter the path to your image directory: ").strip()
    # Can be changed to whatever format to work with GUI

    # Binarize images
    features = process_directory(input_path)

    if features:
        print("Binarization complete.")

        # Extract features from the binarized images
        binarized_path = "binarized_images"
        skeleton_path = "skeletonized_images"
        visuals = False

        setup_extract_features(skeleton_path, binarized_path, input_path, features, visuals)
        print("Features extracted to extracted_features")

    else:
        print("Processing failed. Please check the input path and try again.")
