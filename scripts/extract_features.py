import cv2
import numpy as np
from plantcv import plantcv as pcv
import porespy as porespy
from scipy.stats import linregress


def extract_features(binarized_img, skeleton_pruned, image_name=None):

    binarized_img = np.where(binarized_img > 0, 255, 0).astype(np.uint8)
    skeleton_pruned = np.where(skeleton_pruned > 0, 255, 0).astype(np.uint8)

    # Extract connected components
    num_components, labels = cv2.connectedComponents(binarized_img)

    features = []
    fractal_dims = [] # saving for testing purposes

    # Iterate over the connected components
    for label in range(1, num_components):  
        component_mask = (labels == label).astype(np.uint8)

        # AREA feature
        area = np.sum(binarized_img[component_mask > 0]) // 255

        # PERIMETER feature
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = sum(cv2.arcLength(cnt, True) for cnt in contours)

        # CIRCULARITY (roundness) feature
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

        # extracting component skeleton for further feature extraction
        component_skeleton = cv2.bitwise_and(skeleton_pruned, skeleton_pruned, mask=component_mask)

        # method 1 to find number of projections, I find method 2 below to be more accurate
        branch_pts = pcv.morphology.find_branch_pts(skel_img=component_skeleton)
        num_projections = np.sum(branch_pts > 0) # number of projections

        # extracting PROJECTION features (see analyze_projections)
        num_projs2, proj_lengths, avg_proj_length, max_proj_length = analyze_projections(component_skeleton)

        # NEIGHBORS feature
        neighbors = set()
        kernel = np.ones((70, 70), np.uint8) # need to determine the area to consider for neighbors
        dilated_mask = cv2.dilate(component_mask, kernel, iterations=1)
        neighbors = np.unique(labels[dilated_mask > 0])
        neighbors = neighbors[(neighbors != label) & (neighbors != 0)]  # removing current component and 0 (background)
        # print(f"Neighbors after exclusion: {neighbors}") 
        num_neighbors = len(neighbors)

        # LENGTH/WIDTH RATIO feature
        x, y, width, height = cv2.boundingRect(component_mask)
        length = max(width, height)
        width = min(width, height)
        length_width_ratio = length / width if width > 0 else 0  # Avoid division by zero

        # FRACTAL DIMENSION feature (lines 59-76)
        skel_x, skel_y, skel_width, skel_height = cv2.boundingRect(component_skeleton)
        cropped_mask = component_skeleton[skel_y: skel_y + skel_height, skel_x: skel_x + skel_width]
        # Thought about calculating fractal dimension after cropping around astrocyte
        # contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # mask = np.zeros_like(component_mask, dtype=np.uint8)
        # cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
        # cropped_mask = cv2.bitwise_and(component_mask, component_mask, mask=mask)
        if num_projs2 > 0:
            fractal_dim = calculate_fractal_dimension(cropped_mask) # see function
        else:
            fractal_dim = 0 # astrocyte with no projections is not fractal, so FD does not apply

        if fractal_dim > 1.6: # function returned large fractal dimension for small, low-res images, so getting rid of them
            fractal_dim = -1

        fractal_dims.append(fractal_dim) # testing purposes

        features.append({
            'image_name': image_name,
            'component': label,
            'area': area,
            'perimeter': perimeter,
            'num_projections': num_projections,
            'num_projections_v2': num_projs2,
            'projection_lengths': proj_lengths,
            'avg_projection_length': avg_proj_length,
            'max_projection_length': max_proj_length,
            'circularity': circularity,
            'neighbors': num_neighbors,
            'length_width_ratio': length_width_ratio,
            'fractal_dim': fractal_dim,
            'mask': component_mask
        })

    # plt.imshow(labels)

    # testing fractal dimensions
    positive_fractal_dims = np.array(fractal_dims)[np.array(fractal_dims) > 0]
    if positive_fractal_dims.size > 0:
        print(f"Positive fractal dimension range: {positive_fractal_dims.min()} to {positive_fractal_dims.max()}")
    else:
        print("No positive fractal dimensions available.")

    return features

def analyze_projections(skeleton_component):

    segments, objs = pcv.morphology.segment_skeleton(skel_img=skeleton_component)

    # segment sort seems pretty accurate, run debug to visualize
    # pcv.params.debug = "plot"
    projection_objs, body_objs = pcv.morphology.segment_sort(skel_img=skeleton_component,
                                                  objects=objs)
    # NUMBER OF PROJECTIONS feature
    num_projs = len(projection_objs)

    # getting ALL PROJECTION LENGTHS feature
    pcv.params.sample_label = 'astrocyte'
    labeled_path_img = pcv.morphology.segment_path_length(segmented_img=segments, objects=projection_objs) # objs must be projections only
    proj_lengths = pcv.outputs.observations['astrocyte']['segment_path_length']['value']

    # AVG/MAX PROJECTION LENGTHS FEATURE
    avg_proj_length = np.mean(proj_lengths) if proj_lengths else 0
    max_proj_length = np.max(proj_lengths) if proj_lengths else 0

    return num_projs, proj_lengths, avg_proj_length, max_proj_length

def calculate_fractal_dimension(binarized_img_cropped):
    # boxcount function links:
    # function overview: https://porespy.org/modules/generated/generated/porespy.metrics.boxcount.html#porespy.metrics.boxcount
    # implementation: https://porespy.org/examples/metrics/tutorials/computing_fractal_dim.html
    # source code: https://porespy.org/_modules/porespy/metrics/_funcs.html
    binary_image = (binarized_img_cropped > 0).astype(np.uint8)

    try:
        # Consider optional binning for boxcount function? probably redundant because of way porespy boxcount uses bins
        skel_x, skel_y, skel_width, skel_height = cv2.boundingRect(binarized_img_cropped)
        num_bins = max(10, min(skel_width, skel_height) // 10)
        # print(num_bins)

        result = porespy.metrics.boxcount(binary_image, bins=10)
        sizes = result.size
        counts = result.count
        # print(result.slope)
        # fractal_dimension = np.mean(result.slope) # not the best approach, see 3rd link above

        # FRACTAL DIMENSION feature - based on algorithm
        log_sizes = np.log(sizes)
        log_counts = np.log(counts)
        slope, _, _, _, _ = linregress(log_sizes, log_counts)

        fractal_dimension = -slope

        return fractal_dimension

    except Exception as e: # Doesnt get called often, but could debug
        print(f"Error in boxcount: {e}")
        return -1
