# Astrocyte Image Processing Tool

A comprehensive tool for processing and analyzing astrocyte images, featuring binarization, skeletonization, feature extraction, and visualization capabilities.

## Table of Contents
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Features](#features)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- PyQt6
- OpenCV (cv2)
- NumPy
- Pandas
- Matplotlib
- scikit-image
- scipy
- plantcv
- skan

## Data Preparation

### Image Organization
1. Create a folder named `data` in the root directory
2. Inside the `data` folder, you can organize your images in two ways:

#### Option 1: Single Directory
Place all your .tiff files directly in the `data` folder:
```
data/
    image1.tiff
    image2.tiff
    image3.tiff
```

#### Option 2: Multiple Directories
Create subdirectories for different groups/conditions:
```
data/
    group1/
        image1.tiff
        image2.tiff
    group2/
        image3.tiff
        image4.tiff
```

### Image Requirements
- Images should be in .tiff format
- For nucleus overlay functionality, include both channel 1 (nucleus) and channel 2 (astrocyte) images
- Channel 1 images should have "-ch1" in the filename
- Channel 2 images should have "-ch2" in the filename
- Example naming convention:
  ```
  image1-ch1.tiff  (nucleus channel)
  image1-ch2.tiff  (astrocyte channel)
  ```

## Usage

1. Run the application:
```bash
python main.py
```

2. Using the GUI:
   - Click "Select Folder" to choose your data directory
   - Click "Process Images" to start the analysis
   - The progress bar will show the current processing status
   - Once complete, you can:
     - View feature visualizations in the "Visualizations" tab
     - Download feature data using the download buttons
     - View charts in the "Charts" tab

## Features

### Image Processing
- **Binarization**: Converts grayscale images to binary using advanced thresholding
- **Skeletonization**: Creates skeleton representations of astrocytes
- **Feature Extraction**: Calculates various morphological features

### Extracted Features
- Area
- Perimeter
- Circularity
- Number of Projections
- Total Skeleton Length
- Fractal Dimension
- Projection Lengths (average and maximum)
- Number of Neighbors
- Length/Width Ratio

### Visualization
- **Feature Visualization**: View individual astrocytes with:
  - Original image
  - Enhanced image
  - Binarized image with skeleton overlay
  - Nucleus overlay (if available)
- **Feature Statistics**: View detailed measurements for each feature
- **Charts**: Generate statistical comparisons between groups

## Output Files

The tool creates several output directories:

1. `binarized_images/`: Contains binarized versions of input images
2. `skeletonized_images/`: Contains skeletonized versions of binarized images
3. `extracted_features/`: Contains CSV files with individual astrocyte features
4. `whole_image_features/`: Contains CSV files with aggregated features per image
5. `charts/`: Contains statistical comparison charts
6. `temp/`: Temporary files for visualization

### CSV File Structure
Each feature CSV file contains:
- File name
- Object label
- All extracted features
- Statistical measurements

## Troubleshooting

### Getting Help
If you encounter issues:
1. Check the console output for error messages
2. Verify your data organization follows the required structure
3. Ensure all dependencies are correctly installed

## Credits

This tool was developed by Bonsai.

### Development Team
- Justin Bonner
- Nikhita Guhan
- Esteban Morales
- Ethan Tieu

Special thanks to all contributors for their work on the binarization algorithms, feature extraction methods, and visualization tools. 
