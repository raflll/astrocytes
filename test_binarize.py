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
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from scipy.ndimage import gaussian_filter



def get_tifs():
    return sorted([os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith(".TIFF") or file.endswith(".tiff")])

def binarize_justin(img):
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

def preprocess_images(images):
    """Preprocess all images and their binary versions"""
    processed = []
    for img in images:
        # Create enhanced version
        enhanced = np.clip(img.astype(np.int32) * 5, 0, 255).astype(np.uint8)
        # Create binary versions
        binary_justin = binarize_justin(img)
        binary_new = binarize_new_1(img)
        # Store all versions
        processed.append({
            'enhanced': enhanced,
            'binary_justin': binary_justin,
            'binary_new': binary_new
        })
    return processed

def setup_figure(n_images):
    """Setup the figure with both grid and hover areas"""
    grid_size = int(np.ceil(np.sqrt(n_images)))
    fig = plt.figure(figsize=(15, 10))
    
    # Create main GridSpec
    gs = GridSpec(1, 1)
    
    # Create hover area GridSpec
    hover_gs = GridSpec(1, 3, wspace=0.3)
    
    # Create hover axes
    hover_axes = [fig.add_subplot(hover_gs[0, i]) for i in range(3)]
    for ax in hover_axes:
        ax.set_visible(False)
        ax.axis('off')
    
    # Calculate grid layout
    cols = grid_size
    rows = (n_images - 1) // cols + 1
    
    return fig, hover_axes, gs[0], (rows, cols)

def create_image_grid(grid_gs, processed_images, grid_dims):
    """Create the main image grid"""
    rows, cols = grid_dims
    axes_info = []
    
    for idx, img_data in enumerate(processed_images):
        row = idx // cols
        col = idx % cols
        
        # Create subplot in the grid
        ax = plt.subplot(rows, cols, idx + 1)
        ax.imshow(img_data['enhanced'], cmap='gray', vmin=0, vmax=255)
        ax.axis('off')
        
        # Store axis information
        axes_info.append({
            'axis': ax,
            'index': idx
        })
    
    return axes_info

def update_hover_display(hover_axes, processed_images, index, visible=True):
    """Update the hover display area"""
    titles = ['Original', 'Binarized (Justin)', 'Binarized (New)']
    images = [
        processed_images[index]['enhanced'],
        processed_images[index]['binary_justin'],
        processed_images[index]['binary_new']
    ] if visible else [None, None, None]
    
    for ax, img, title in zip(hover_axes, images, titles):
        ax.clear()
        if visible and img is not None:
            ax.imshow(img, cmap='gray', vmin=0, vmax=255)
            ax.set_title(title, fontsize=12, pad=10)
        ax.set_visible(visible)
        ax.axis('off')
    
    plt.draw()

def handle_hover(event, hover_axes, processed_images, axes_info, current_hover, fig):
    """Handle hover events"""
    if event.inaxes:
        # Find if we're hovering over an image in the grid
        for info in axes_info:
            if event.inaxes == info['axis']:
                if current_hover != info['index']:
                    # Create frosted pane if it doesn't exist
                    if not hasattr(fig, 'frost'):
                        # Generate frosted pane
                        frost = np.random.random((fig.canvas.get_width_height()[::-1] + (3,))) * 255
                        frost = gaussian_filter(frost, sigma=10)
                        frost = (frost * 0.5).astype(np.uint8)
                        fig.frost = fig.add_axes([0, 0, 1, 1], zorder=10)
                        fig.frost.imshow(frost, extent=[0, 1, 0, 1], transform=fig.transFigure, alpha=0.7)
                        fig.frost.axis('off')
                    
                    # Show frosted pane and hover images
                    fig.frost.set_visible(True)
                    for ax in hover_axes:
                        ax.set_zorder(20)
                        ax.set_visible(True)
                    
                    # Update hover display
                    update_hover_display(hover_axes, processed_images, info['index'], True)
                    return info['index']
                else:
                    # Keep frosted pane and hover images visible while hovering
                    return current_hover
        
        # Check if we're hovering over the hover display
        if event.inaxes in hover_axes:
            return current_hover
    
    # If we're not over any relevant axes, hide the hover display and frosted pane
    if current_hover is not None:
        if hasattr(fig, 'frost'):
            fig.frost.set_visible(False)
        update_hover_display(hover_axes, processed_images, current_hover, False)
    return None

def display_interactive_images(images):
    # Preprocess all images
    processed_images = preprocess_images(images)
    
    # Setup the figure and axes
    fig, hover_axes, grid_gs, grid_dims = setup_figure(len(images))
    
    # Create the image grid
    axes_info = create_image_grid(grid_gs, processed_images, grid_dims)
    
    # Initialize hover state
    current_hover = None
    
    # Create hover event handler
    def on_hover(event):
        nonlocal current_hover
        current_hover = handle_hover(event, hover_axes, processed_images, axes_info, current_hover, fig)
    
    # Connect event handler
    fig.canvas.mpl_connect('motion_notify_event', on_hover)
    
    plt.show()


input_dir = 'test'
images = get_tifs()
images = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in images]
display_interactive_images(images)