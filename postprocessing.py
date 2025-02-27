import copy
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import pandas as pd
import math

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

def setup_extract_features(skeletonized, binarized, data, visual):
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
            extract_features(skeletonized_subfolder_path, binarized_images,
                             data_subfolder_path, visual)

def extract_features(skeleton_images, binarized_images, data_subfolder_path, visual):
    # Create a dictionary to store features by subfolder
    subfolder_features = {}

    for i, binarized_image_path in enumerate(binarized_images):
        # Get relative path for feature lookup
        image_filename = os.path.basename(binarized_image_path)
        subfolder = os.path.basename(os.path.dirname(binarized_image_path))
        feature_key = os.path.join(subfolder, image_filename)

        data_image_path = os.path.join(data_subfolder_path, image_filename)
        skeleton_image_path = os.path.join(skeleton_images, image_filename)

        if os.path.exists(data_image_path) and os.path.exists(skeleton_image_path):
            # Load images
            binarized_image = cv2.imread(binarized_image_path, cv2.IMREAD_GRAYSCALE)
            data_image = cv2.imread(data_image_path, cv2.IMREAD_GRAYSCALE)
            skeleton_image = cv2.imread(skeleton_image_path, cv2.IMREAD_GRAYSCALE)

            # Count components
            num_labels, labels = cv2.connectedComponents(binarized_image)

            # Get features for this image from CSV files
            feature_stats = get_features(image_filename)
            if feature_stats:
                # Add image filename to the features
                feature_stats['image_filename'] = image_filename

                # Initialize list for this subfolder if it doesn't exist
                if subfolder not in subfolder_features:
                    subfolder_features[subfolder] = []

                # Add features to the appropriate subfolder list
                subfolder_features[subfolder].append(feature_stats)

            # Visualization code
            if visual:
                plt.figure(figsize=(10, 5))
                data_image = cv2.cvtColor(data_image, cv2.COLOR_GRAY2BGR)
                binarized_image = cv2.cvtColor(binarized_image, cv2.COLOR_GRAY2BGR)

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

                plt.show()

        else:
            print(f"Matching file not found in data for: {image_filename}")
    # Save features to CSV files for each subfolder
    save_features_to_csv(subfolder_features)

def save_features_to_csv(subfolder_features):
    # Save extracted features to csv for subfolder
    for subfolder, features_list in subfolder_features.items():
        if features_list:
            # Convert list of features to DataFrame
            df = pd.DataFrame(features_list)

            # Create output directory if it doesn't exist
            output_dir = 'whole_image_features'
            os.makedirs(output_dir, exist_ok=True)

            # Save to CSV
            csv_filename = os.path.join(output_dir, f'{subfolder}_features.csv')
            df.to_csv(csv_filename, index=False)
            print(f"Saved features for {subfolder} to {csv_filename}")

def get_features(image_path):
    """
    Calculate aggregate features for the given image by reading from CSV files.

    Parameters:
    image_path (str): The image filename to look for in the CSV files

    Returns:
    dict: Calculated feature averages for the image
    """
    # CSV file paths
    csv_files = {
        "Phenotype 1": "extracted_features/Phenotype 1_features.csv",
        "Phenotype 2": "extracted_features/Phenotype 2_features.csv",
        "Control": "extracted_features/Control_features.csv",
        "Images": "extracted_features/Image_features.csv"
    }

    # Search for the image in all CSV files
    astrocyte_features = []

    for csv_name, csv_path in csv_files.items():
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                # Filter rows that match the image filename
                image_data = df[df['file_name'] == image_path]

                if not image_data.empty:
                    # Convert each row to a dictionary for processing
                    for _, row in image_data.iterrows():
                        # Convert branch_lengths and projection_lengths from string to list if they exist
                        feature_dict = row.to_dict()

                        # Convert string representations of lists to actual lists
                        for list_col in ['branch_lengths', 'projection_lengths']:
                            if list_col in feature_dict and isinstance(feature_dict[list_col], str):
                                try:
                                    # Handle list format (ex: "[1.2, 3.4, 5.6]")
                                    feature_dict[list_col] = eval(feature_dict[list_col])
                                except:
                                    # If eval fails, provide an empty list
                                    feature_dict[list_col] = []

                        astrocyte_features.append(feature_dict)

                    break  # Found the image, no need to check other CSV files

            except Exception as e:
                print(f"Error reading {csv_path}: {e}")

    if not astrocyte_features:
        return None

    # Calculate aggregate statistics
    Totals = {
        "num_branches": 0,
        "branch_lengths": 0,
        "total_skeleton_length": 0,
        "most_branches": 0,
        "analyzed": 0,
        "perimeter": 0,
        "roundness": 0,
        "average_area": 0,
        "largest_area": 0,
        "average_perimeter": 0,
        "largest_perimeter": 0,
        "circularity": 0,
        "roundness_variance": 0,
        "fractal_dim": 0,
        "num_projections": 0,
        "num_neighbors": 0,
        "proj_length": 0,
        "most_neighbors": 0,
        "most_projections": 0
    }

    for f in astrocyte_features:  # Iterate through features for each astrocyte
        Totals["num_branches"] += f.get("num_branches", 0)

        # Handle branch_lengths (could be a list or not exist)
        if "branch_lengths" in f and isinstance(f["branch_lengths"], list):
            Totals["branch_lengths"] += sum(f["branch_lengths"])

        Totals["total_skeleton_length"] += f.get("total_skeleton_length", 0)
        Totals["most_branches"] = max(Totals["most_branches"], f.get("num_branches", 0))
        Totals["analyzed"] += 1
        Totals["perimeter"] += f.get("perimeter", 0)
        Totals["roundness"] += f.get("roundness", 0)
        Totals["average_area"] += f.get("area", 0)
        Totals["largest_area"] = max(Totals["largest_area"], f.get("area", 0))
        Totals["average_perimeter"] += f.get("perimeter", 0)
        Totals["largest_perimeter"] = max(Totals["largest_perimeter"], f.get("perimeter", 0))
        Totals["circularity"] += f.get("circularity", 0)
        Totals["fractal_dim"] += f.get("fractal_dim", 0)
        Totals["num_projections"] += f.get("num_projections", 0)

        # Handle projection_lengths (could be a list or not exist)
        if "projection_lengths" in f and isinstance(f["projection_lengths"], list):
            Totals["proj_length"] += sum(f["projection_lengths"])

        Totals["num_neighbors"] += f.get("neighbors", 0)
        Totals["most_neighbors"] = max(Totals["most_neighbors"], f.get("neighbors", 0))
        Totals["most_projections"] = max(Totals["most_projections"], f.get("num_projections", 0))

    # Calculate averages (avoid division by zero)
    bl = Totals["branch_lengths"] / Totals["num_branches"] if Totals["num_branches"] > 0 else 0
    sl = Totals["total_skeleton_length"] / Totals["analyzed"] if Totals["analyzed"] > 0 else 0
    ar = Totals["roundness"] / Totals["analyzed"] if Totals["analyzed"] > 0 else 0
    aa = Totals["average_area"] / Totals["analyzed"] if Totals["analyzed"] > 0 else 0
    ap = Totals["average_perimeter"] / Totals["analyzed"] if Totals["analyzed"] > 0 else 0
    bd = (Totals["num_branches"] / Totals["analyzed"] / sl) if (Totals["analyzed"] > 0 and sl > 0) else 0
    ab = Totals["num_branches"] / Totals["analyzed"] if Totals["analyzed"] > 0 else 0
    ac = Totals["circularity"] / Totals["analyzed"] if Totals["analyzed"] > 0 else 0
    anp = Totals["num_projections"] / Totals["analyzed"] if Totals["analyzed"] > 0 else 0
    afd = Totals["fractal_dim"] / Totals["analyzed"] if Totals["analyzed"] > 0 else 0
    nn = Totals["num_neighbors"] / Totals["analyzed"] if Totals["analyzed"] > 0 else 0
    pl = Totals["proj_length"] / Totals["analyzed"] if Totals["analyzed"] > 0 else 0

    # Calculate roundness variance
    for f in astrocyte_features:
        Totals["roundness_variance"] += (f.get("roundness", 0) - ar)**2

    rv = Totals["roundness_variance"] / (Totals["analyzed"] - 1) if Totals["analyzed"] > 1 else 0

    Averages = {
        "num_branches": f"{bl:.3f}",
        "branch_lengths": f"{bl:.3f}",
        "skeleton_length": f"{sl:.3f}",
        "most_branches": Totals["most_branches"],
        "analyzed": Totals["analyzed"],
        "average_roundness": f"{ar:.3f}",
        "average_area" : f"{aa:.3f}",
        "a_ratio" : f"{aa / Totals['largest_area']:.3f}" if Totals['largest_area'] > 0 else "0.000",
        "average_perimeter" : f"{ap:.3f}",
        "largest_perimeter" : Totals["largest_perimeter"],
        "thickness" : f"{aa / sl:.3f}" if sl > 0 else "0.000",
        "branch_density" : f"{bd:.3f}",
        "average_branches" : f"{ab:.3f}",
        "average_circularity" : f"{ac:.3f}",
        "roundness_variance" : f"{rv:.3f}",
        "average_fractal_dim" : f"{afd:.3f}",
        "average_num_projections" : f"{anp:.3f}",
        "num_neighbors" : f"{nn:.3f}",
        "proj_length": f"{pl:.3f}",
        "most_neighbors": Totals["most_neighbors"],
        "most_projections": Totals["most_projections"]
    }

    return Averages
