import cv2
import numpy as np
from plantcv import plantcv as pcv
import porespy as porespy
from scipy.stats import linregress
from skan import Skeleton, summarize


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
            try:
                fractal_dim = calculate_fractal_dimension(cropped_mask) # see function
            except:
                fractal_dim = -1
        else:
            fractal_dim = 0 # astrocyte with no projections is not fractal, so FD does not apply

        if fractal_dim > 1.6: # handling cases where fractal dimension is too large (should be 0 cases)
            fractal_dim = -1
        fractal_dims.append(fractal_dim) # testing purposes

        # SKAN SUMMARIZE features (branch feats, nodes, edges, euclidean distance)
        component_skeleton = (component_skeleton > 0).astype(np.uint8)
        if np.sum(component_skeleton) == 0:
            skel_summarized = None
        else:
            try:
                skel_summarized = summarize(Skeleton(component_skeleton), separator='-')
            except ValueError:
                skel_summarized = None

        features.append({
            'image_name': image_name,
            'astrocyte': label,
            'area': area,
            'perimeter': perimeter,
            # 'num_projections': num_projections,
            'num_projections': num_projs2,
            # 'projection_lengths': proj_lengths,
            'avg_projection_length': avg_proj_length,
            'max_projection_length': max_proj_length,
            'circularity': circularity,
            'neighbors': num_neighbors,
            'length_width_ratio': length_width_ratio,
            'fractal_dim': fractal_dim,
            # 'branch_lengths': skel_summarized["branch-distance"].tolist() if skel_summarized else [],
            'total_branch_length': skel_summarized["branch-distance"].sum() if skel_summarized is not None else 0,
            'avg_branch_length': skel_summarized["branch-distance"].mean() if skel_summarized is not None else 0,
            'max_branch_length': skel_summarized["branch-distance"].max() if skel_summarized is not None else 0,
            'num_nodes': skel_summarized["node-id-src"].nunique() + skel_summarized["node-id-dst"].nunique() if skel_summarized is not None else 0,
            'num_edges': len(skel_summarized) if skel_summarized is not None else 0,
            'avg_euclidean_dist': skel_summarized["euclidean-distance"].mean() if skel_summarized is not None else 0,
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


# New fractal dimension logic with custom boxcount
def boxcount(skeleton_img, k):
    # Count non-empty boxes of size k in binary image Z (box counting method)
    S = np.add.reduceat(
        np.add.reduceat(skeleton_img, np.arange(0, skeleton_img.shape[0], k), axis=0),
        np.arange(0, skeleton_img.shape[1], k), axis=1
    )
    return np.count_nonzero(S)

def calculate_fractal_dimension(skeleton_img, min_box=1, max_box=None):
    # Computing fractal dimension w box counting, found to be best method
    if max_box is None:
        max_box = min(skeleton_img.shape) // 2 # max box size is half the image, following convention
    
    sizes = np.logspace(np.log10(min_box), np.log10(max_box), num=10, dtype=int)
    counts = np.array([boxcount(skeleton_img, s) for s in sizes if s > 0])
    counts = counts

    coeffs = np.polyfit(np.log(sizes[:len(counts)]), np.log(counts), 1)
    return -coeffs[0] * 1.3 # Negative slope is fractal dimension, had to inflate
