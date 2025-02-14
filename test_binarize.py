import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tifffile import imsave
from scipy.integrate import quad
import skimage as ski
from scipy.signal import find_peaks
from skimage.filters import unsharp_mask
from scipy import ndimage
from skimage import feature
import skimage.filters as filters
from scipy.ndimage import label



def get_tifs():
    return sorted([os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith(".TIFF") or file.endswith(".tiff")])

def binarize_justin(img):
    # TODO: Make SIZE_FILTER dynamic depending on the image? I'm not sure if this is possible or necessary
    SIZE_FILTER = 75 # Lower if we are not detecting small cells, raise if we are getting noise

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

    return binary

def binarize_new_1(image):

    thresh = ski.filters.threshold_triangle(image)
    binary_full = image > thresh
    binary_full = (binary_full * 255).astype(np.uint8)

    # Define the kernel for dilation
    kernel = np.ones((5, 5), np.uint8)

    binary_full = cv2.morphologyEx(binary_full, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(binary_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask to store the filtered particles
    final_mask = np.zeros_like(binary_full)

    # Loop through contours and keep only large ones
    for cnt in contours:
        contour_mask = np.zeros_like(binary_full)
        cv2.drawContours(contour_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        # Dilate the contour
        dilated_mask = cv2.dilate(contour_mask, kernel, iterations=1)

        # Find bounding rectangle of the dilated mask
        x, y, w, h = cv2.boundingRect(dilated_mask)

        # Crop the original image to the bounding box
        cropped_region = image[y:y+h, x:x+w]
        cropped_dilated_mask = dilated_mask[y:y+h, x:x+w]

        enhance_contrast = 10
        cropped_region = np.clip(cropped_region.astype(np.int32) * enhance_contrast, 0, 255).astype(np.uint8)

        # Apply triangle thresholding to the masked region
        block_size = w//2
        if block_size%2 == 0:
            block_size = block_size + 1
        thresh = ski.filters.threshold_local(cropped_region, block_size=block_size)
        binary_region = cropped_region > thresh
        binary_region = (binary_region * 255).astype(np.uint8)

        # Label connected components in binary_region
        labeled_region, num_features = label(binary_region)

        # Create an empty array to store the resulting components
        kept_components = np.zeros_like(binary_region)

        # Iterate through each component and check for overlap with cropped_dilated_mask
        for i in range(1, num_features + 1):
            # Create a mask for the current component
            component_mask = (labeled_region == i)

            # Check if there is any overlap with cropped_dilated_mask
            if np.any(component_mask & cropped_dilated_mask):  # If overlap exists
                kept_components[component_mask] = 255  # Keep the component

        # Add the thresholded cropped region to the final mask at the correct location
        final_mask[y:y+h, x:x+w] = cv2.bitwise_or(final_mask[y:y+h, x:x+w], kept_components)

    # Fill holes
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Compute sizes (areas) of all detected particles
    particle_sizes = np.array([cv2.contourArea(cnt) for cnt in contours])
    size_thresh = ski.filters.threshold_triangle(particle_sizes)

    # Initialize the output mask
    output_mask = np.zeros_like(binary_full)

    # Draw contours that have an area above the threshold
    for cnt in contours:
        if cv2.contourArea(cnt) > size_thresh:  # Only draw if the area is above the threshold
            cv2.drawContours(output_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    return output_mask

def binarize_new_2(image):

    # binary = np.clip(image.astype(np.int32) * 10, 0, 255).astype(np.uint8)
    
    # Find the most frequent pixel intensity
    unique_colors, counts = np.unique(image, return_counts=True)
    background_color = unique_colors[np.argmax(counts)]  # Color with max occurrences

    # Set all pixels with this value to 0
    denoised_image = np.where(image <= background_color, 0, image)

    masked_image = np.where(image > background_color, 255, image)
    masked_image = cv2.morphologyEx(masked_image, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))

    # Find contours
    contours, _ = cv2.findContours(masked_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Compute sizes (areas) of all detected particles
    particle_sizes = np.array([cv2.contourArea(cnt) for cnt in contours])
    size_thresh = ski.filters.threshold_triangle(particle_sizes)

    # Create a mask to store the filtered particles
    final_mask = np.zeros_like(image)

    # Define the kernel for dilation
    kernel = np.ones((5, 5), np.uint8)

    # Loop through contours and keep only large ones
    for cnt in contours:
        if cv2.contourArea(cnt) >= size_thresh:
            contour_mask = np.zeros_like(image)
            cv2.drawContours(contour_mask, [cnt], -1, 255, thickness=cv2.FILLED)

            # Find bounding rectangle of the dilated mask
            x, y, w, h = cv2.boundingRect(contour_mask)

            # Crop the original image to the bounding box
            cropped_region = denoised_image[y:y+h, x:x+w]

            # Crop the dilated mask to the bounding box
            cropped_dilated_mask = contour_mask[y:y+h, x:x+w]

            # Apply triangle thresholding to the masked region
            thresh = ski.filters.threshold_sauvola(cropped_region)
            binary_region = cropped_region > thresh
            binary_region = (binary_region * 255).astype(np.uint8)

            best_component = binary_region
           
            final_mask[y:y+h, x:x+w] = cv2.bitwise_or(final_mask[y:y+h, x:x+w], best_component)

    return masked_image

def binarize_new_3(orig_image):
    image = orig_image
    unique_colors, counts = np.unique(image, return_counts=True)
    
    mult = 2

    while len(unique_colors) > 2 and mult < 14:
        # Ignore pure black (0) and pure white (255), find the most frequent unwanted color
        filtered_colors = unique_colors[(unique_colors != 0) & (unique_colors != 255)]
        
        if len(filtered_colors) == 0:
            break  # No more unwanted colors

        # Find the most frequent non-(0,255) color
        background_color = filtered_colors[np.argmax(counts[np.isin(unique_colors, filtered_colors)])]

        image = np.clip(orig_image.astype(np.int32) * mult, 0, 255).astype(np.uint8)

        mult = mult + 1

        # Set pixels <= background_color to 0
        image[image == background_color] = 0
        
        # Recalculate unique colors
        unique_colors, counts = np.unique(image, return_counts=True)

    return image

def binarize_new_4(image):
    unique_colors, counts = np.unique(image, return_counts=True)
    background_color = unique_colors[np.argmax(counts)]
    denoised_image = np.where(image <= background_color, 0, image)

    max_color = np.max(unique_colors)
    mult = np.ceil(255/max_color)

    denoised_image = np.clip(denoised_image.astype(np.int32) * mult, 0, 255).astype(np.uint8)

    # image = np.where(image == 255, 255, 0)

    thresh = ski.filters.threshold_local(image, 33)
    image = denoised_image > thresh
    image = (image * 255).astype(np.uint8)

    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))

    # unique_colors, counts = np.unique(denoised_image, return_counts=True)

   # Exclude color 0
    # mask = unique_colors != 0
    # filtered_colors = unique_colors[mask]
    # filtered_counts = counts[mask]

    # # Plot histogram
    # plt.bar(filtered_colors, filtered_counts, color='gray', edgecolor='black')
    # plt.xlabel("Pixel Intensity")
    # plt.ylabel("Frequency")
    # plt.title("Histogram of Unique Colors (Excluding 0)")
    # plt.xticks(filtered_colors)  # Ensure all unique values appear on x-axis
    # plt.show()

    
    
    return image

def analyze_image_histogram(image):
    # Calculate histogram
    hist, bin_edges = np.histogram(image.ravel(), bins=100, range=(0, 256))
    
    # # Find peaks
    peaks, _ = find_peaks(hist, height=max(hist)*0.001)  # Adjust threshold as needed
    
    # # Find leftmost and rightmost peaks
    left_peak = peaks[0]
    right_peak = peaks[-1]
    
    # # Find minimum between peaks
    valley_region = hist[left_peak:right_peak]
    valley_position = left_peak + np.argmin(valley_region)
    
    # # Create the plot
    plt.figure(figsize=(12, 6))
    
    # # Plot histogram
    plt.hist(image.ravel(), bins=100, range=(0,50), color='gray', alpha=0.7)

    # print(image.ravel())
    
    # Plot vertical lines at peaks
    plt.axvline(x=left_peak, color='red', linestyle='--', 
                label=f'Left Peak (value={left_peak})')
    plt.axvline(x=right_peak, color='blue', linestyle='--', 
                label=f'Right Peak (value={right_peak})')
    
    # # Plot valley line
    plt.axvline(x=valley_position, color='green', linestyle='--', 
                label=f'Valley (value={valley_position})')
    
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.title('Image Histogram with Peak and Valley Detection')
    plt.legend()
    
    return

input_dir = 'test'
images = get_tifs()
images = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in images][2:4]
# analyze_image_histogram(images[1])

# Create the plot
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
plt.subplots_adjust(hspace=0, wspace=0.1) 

# Set column titles
axes[0, 0].set_title('Original Images')
axes[0, 1].set_title('Binarized Images Justin')
axes[0, 2].set_title('Binarized Images New')

for i, image in enumerate(images):
    # analyze_image_histogram(image)
    # Original image
    axes[i, 0].imshow(np.clip(image.astype(np.int32) * 5, 0, 255).astype(np.uint8), cmap='gray', vmin=0, vmax=255)
    axes[i, 0].axis('off')

    # Binarized image old
    bin_image_old = binarize_justin(image)
    axes[i, 1].imshow(bin_image_old, cmap='gray', vmin=0, vmax=255)
    axes[i, 1].axis('off')

    # Binarized image new
    bin_image_new = binarize_new_1(image)
    axes[i, 2].imshow(bin_image_new, cmap='gray', vmin=0, vmax=255)
    axes[i, 2].axis('off')

plt.show()
