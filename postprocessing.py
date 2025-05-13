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

def get_features(image_name):
    """Get features for a specific image from all CSV files"""
    features = {}
    
    # Check if extracted_features directory exists
    if not os.path.exists('extracted_features'):
        print("Warning: extracted_features directory not found!")
        return features
    
    # Get all CSV files in the extracted_features directory
    csv_files = [f for f in os.listdir('extracted_features') if f.endswith('_features.csv')]
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(os.path.join('extracted_features', csv_file))
            
            # Find the row with matching image name
            matching_rows = df[df['file_name'] == image_name]
            
            if not matching_rows.empty:
                # Get the first matching row
                row = matching_rows.iloc[0]
                
                # Convert string representations of lists to actual lists
                for col in ['projection_lengths']:
                    if col in row and isinstance(row[col], str):
                        try:
                            row[col] = eval(row[col])
                        except:
                            row[col] = []
                
                # Store features
                features = {
                    'file_name': row['file_name'],
                    'object_label': row['object_label'],
                    'area': row['area'],
                    'perimeter': row['perimeter'],
                    'circularity': row['circularity'],
                    'num_projections': row['num_projections'],
                    'total_skeleton_length': row['total_skeleton_length'],
                    'fractal_dim': row['fractal_dim'],
                    'projection_lengths': row['projection_lengths'],
                    'avg_projection_length': row['avg_projection_length'],
                    'max_projection_length': row['max_projection_length'],
                    'neighbors': row['neighbors'],
                    'length_width_ratio': row['length_width_ratio']
                }
                break
                
        except Exception as e:
            print(f"Error reading {csv_file}: {str(e)}")
            continue
    
    return features
