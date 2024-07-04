import time

import cv2
import argparse
import warnings
import numpy as np
import igraph as ig
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

GC_BGD = 0  # Hard bg pixel
GC_FGD = 1  # Hard fg pixel, will not be used
GC_PR_BGD = 2  # Soft bg pixel
GC_PR_FGD = 3  # Soft fg pixel

HARD_BG = GC_BGD
HARD_FG = GC_FGD
SOFT_BG = GC_PR_BGD
SOFT_FG = GC_PR_FGD

# convergence parameters
EPSILON = 1e-4
CONVERGE = 100
PREV_ENERGY = 0
LAMBDA = 1

# define the eight directions for neighbors
EIGHT_DIR = np.array([(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)])  # bidirectional way


# EIGHT_DIR = np.array([(0, 1), (-1, 1), (-1, 0), (-1, -1)]) # one directional way


# Define the GrabCutResult algorithm function
def grabcut(img: np.ndarray, rect: tuple[int, ...], n_iter: int = 30, gmm_components: int = 5):
    # Assign initial labels to the pixels based on the bounding box
    # Initialize all cells as Background
    global PREV_ENERGY
    mask = np.zeros(img.shape[:2], dtype=np.uint8) * HARD_BG
    x, y, w, h = rect

    # Convert from absolute coordinates
    w -= x
    h -= y

    # Initialize the inner square to Foreground
    mask[y:y + h, x:x + w] = SOFT_FG
    # Initialize center pixel to Foreground - Strong. deleted since not documented
    # mask[rect[1] + rect[3] // 2, rect[0] + rect[2] // 2] = HARD_FG

    # todo: updateGMM is immediately after init - maybe add into update GMM with option to set the GMMs as None -> Init
    bgGMM, fgGMM = initalize_GMMs(img, mask, gmm_components)
    s = time.process_time()
    beta = calculate_beta(img=img)
    print(f"beta time = {time.process_time() - s}")
    s = time.process_time()

    N_link_weights = LAMBDA * calculate_N_link_weights(img=img, beta=beta)
    print(f"N_links calc time = {time.process_time() - s}")
    s = time.process_time()
    N_link_edges, N_link_capacities = format_n_links(N_link_weights, img.shape[0], img.shape[1])
    print(f"N_links format time = {time.process_time() - s}")
    for i in range(n_iter):
        print(i)
        s = time.process_time()
        if i != 0:
            bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM, gmm_components)
        print(f"update gmm time = {time.process_time() - s}")
        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM, N_link_edges, N_link_capacities)

        s = time.process_time()
        mask = update_mask(mincut_sets, mask)
        print(f"update mask time = {time.process_time() - s}")
        s = time.process_time()
        if check_convergence(energy - PREV_ENERGY):
            print(f'Converged in {i + 1}/{n_iter} iterations')
            break
        print(f"convergnece time = {time.process_time() - s}")
        PREV_ENERGY = energy

    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM


def get_trimaps(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    helper function: extract foreground and background pixels from image.
    :param img:  (H x W, C) flatten RGB image.
    :param mask: (H' x W') rectangle representing the (inside) foreground and (outside) background pixels.
    :return: (H' x W', C) foreground and background tree maps.
    """
    mask = mask.reshape(-1)
    # boolean arrays holding the foreground and background pixels masks.
    fore_mask = np.isin(mask, test_elements=[SOFT_FG]).reshape(-1)
    back_mask = np.isin(mask, test_elements=[HARD_BG, SOFT_BG]).reshape(-1)

    fore_trimap = img[fore_mask]
    back_trimap = img[back_mask]

    return fore_trimap, back_trimap


def initalize_GMMs(img: np.ndarray, mask: np.ndarray, n_components: int = 5) -> tuple[GaussianMixture, GaussianMixture]:
    """
    Initialized Gaussian Mixture Models required for the GrabCutResult algorithm.
    :param img: (H, W, C) RGB image.
    :param mask: (H' x W', C) rectangle representing the (inside) foreground and (outside) background pixels.
    :param n_components: number of Gaussian mixtures to create.
    :return: initialized foreground and background Gaussian Mixture Models.
    """
    fore_trimap, back_trimap = get_trimaps(img=img.reshape(-1, 3), mask=mask)

    # finds n means of foreground and background RGB tree maps.
    fore_kmeans = KMeans(n_clusters=n_components, random_state=0).fit(fore_trimap)
    back_kmeans = KMeans(n_clusters=n_components, random_state=0).fit(back_trimap)

    # init n GMM components with n clusters for foreground and background
    fgGMM = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
    fgGMM.means_ = fore_kmeans.cluster_centers_
    # todo: change fit to manual model update
    fgGMM.fit(fore_trimap)

    bgGMM = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
    bgGMM.means_ = back_kmeans.cluster_centers_
    # todo: change fit to manual model update
    bgGMM.fit(back_trimap)

    return bgGMM, fgGMM


def update_parameters(img: np.ndarray, gmm: GaussianMixture, n_components: int = 5) -> GaussianMixture:
    """
    helper function: computes weights, means, and covariances for the Gaussian Mixture Models,
     then update them respectively.
    :param img: (H, W , C) RGB image.
    :param gmm: Gaussian Mixture Model.
    :param n_components: number of Gaussian mixtures to create.
    :return: updated Gaussian Mixture Model (mean, weights, covariances).
    """
    # evaluate the components' density for each sample.
    predictions = gmm.predict_proba(img.reshape(-1, img.shape[-1]))

    weights = np.sum(predictions, axis=0) + EPSILON

    # means
    means = np.dot(predictions.T, img.reshape(-1, img.shape[-1])) / weights[:, np.newaxis]

    # covariance
    covariances = []
    for k in range(n_components):
        diff = img.reshape(-1, img.shape[-1]) - means[k]
        cov = np.dot((predictions[:, k][:, np.newaxis] * diff).T, diff) / weights[k]
        cov += np.eye(cov.shape[0]) * EPSILON  # ensure that cov does not contain 0 on main diagonal -> inv is positive
        covariances.append(cov)

    weights /= weights.sum()

    # model update
    gmm.means_ = means
    gmm.covariances_ = covariances
    gmm.weights_ = weights
    gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covariances))

    return gmm


def update_GMMs(img: np.ndarray,
                mask: np.ndarray,
                bgGMM: GaussianMixture,
                fgGMM: GaussianMixture,
                n_components: int = 5) -> tuple[GaussianMixture, GaussianMixture]:
    """
    updates foreground and background Gaussian Mixture Models from a new mask.
    :param n_components: n_components.
    :param img: (H, W, C) RGB image.
    :param mask: (H' x W', C) rectangle representing the (inside) foreground and (outside) background pixels.
    :param bgGMM: background Gaussian Mixture Model.
    :param fgGMM: foreground Gaussian Mixture Model.
    :return: updated foreground and background Gaussian Mixture Models.
    """
    fore_trimap, back_trimap = get_trimaps(img=img.reshape(-1, 3), mask=mask)

    bgGMM = update_parameters(img=back_trimap, gmm=bgGMM, n_components=n_components)
    fgGMM = update_parameters(img=fore_trimap, gmm=fgGMM, n_components=n_components)

    return bgGMM, fgGMM


def squared_color_difference(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """
    helper function: computes squared difference between original image and shifted (by dx, dy) image.
    ignores the border. Implemented with Vectorization.
    :param img: (H, W, C) RGB image.
    :param dx: shift along x-axis.
    :param dy: shift along y-axis.
    :return: difference between original image and shifted (by dx, dy) image.
    """
    # compute the shifted images
    shifted_img = np.roll(img, shift=(dy, dx), axis=(0, 1))
    # compute the squared differences
    squared_diff = np.sum((img - shifted_img) ** 2, axis=2)
    return squared_diff


def calculate_beta(img: np.ndarray) -> float:
    """
    Calculate the β normalization term for N-link weights.
    :param img: (H, W, C) RGB image.
    :return: β. ensures that the exponential term switches appropriately between high and low contrast.
    """
    height, width, _ = img.shape
    sum_squared_diff = 0

    # iterate through each direction
    for dy, dx in EIGHT_DIR:
        squared_diff = squared_color_difference(img=img, dx=dx, dy=dy)
        sum_squared_diff += np.sum(squared_diff)

    # normalize beta
    beta = sum_squared_diff / (2 * height * width * 8)

    return 1 / (2 * beta)


def format_n_links(N_link_weights, h, w):
    pixel_indices = np.arange(h * w).reshape(h, w)

    # Initialize lists to accumulate neighbor edges and capacities
    neighbor_edges_list = []
    neighbor_capacities_list = []
    for i, (dy, dx) in enumerate(EIGHT_DIR):
        ny, nx = np.meshgrid(np.arange(h) + dy, np.arange(w) + dx, indexing='ij')
        valid_mask = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)

        pixel_indices_flat = pixel_indices[valid_mask]
        neighbor_indices_flat = pixel_indices[ny[valid_mask], nx[valid_mask]]

        neighbor_edges = np.column_stack((pixel_indices_flat, neighbor_indices_flat))
        neighbor_capacities = N_link_weights[:, :, i][valid_mask]

        # Append the results to the lists
        neighbor_edges_list.append(neighbor_edges)
        neighbor_capacities_list.append(neighbor_capacities)

    # Concatenate all the accumulated results
    neighbor_edges = np.concatenate(neighbor_edges_list, axis=0)
    neighbor_capacities = np.concatenate(neighbor_capacities_list, axis=0)

    return neighbor_edges, neighbor_capacities


def calculate_N_link_weights(img: np.ndarray, beta: float) -> np.ndarray:
    """
    Calculate the weights between neighboring pixels, computed using the intensity differences and the β value.
    Follows each pixel's eight neighbors and assigns a weighted edge between them.
    :param img: RGB image (H x W x C)
    :param beta: β is used to determine the weight of N-links
    :return: A NumPy array containing the weights of N-links. Shape is (height, width, 8).
    """
    height, width, _ = img.shape

    # define the eight directions inverse_distances
    distances = np.linalg.norm(EIGHT_DIR, axis=1)
    inverse_distances = 1 / distances

    # Calculate squared differences for all directions
    squared_diffs = np.array([squared_color_difference(img, dx, dy) for dy, dx in EIGHT_DIR])

    # Calculate weights using vectorized operations
    weights = np.exp(-beta * squared_diffs) * 50 * inverse_distances[:, np.newaxis, np.newaxis]

    # Initialize and store weights in N_link_weights array
    N_link_weights = np.transpose(weights, (1, 2, 0))
    return N_link_weights


def calculate_likelihood(img: np.ndarray, gmm: GaussianMixture) -> np.ndarray:
    """
    Accumulates the sum of the weighted Gaussian probabilities for each pixel,
    and then we take the negative logarithm of this sum to get D(m).
    :param img: (H, W, C)
    :param gmm: Gaussian Mixture Model.
    :return: (H, W) - Likelihood for each pixel
    """
    h, w, c = img.shape
    pixels = img.reshape(-1, 3)
    # todo: dont hate me, but is this allowed? or should we re-write the entire formula?
    likelihoods = np.zeros((h, w))

    for i in range(gmm.n_components):
        mean = gmm.means_[i]
        cov = gmm.covariances_[i]
        weight = gmm.weights_[i]
        inv_cov = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)

        norm_factor = weight / np.sqrt(det_cov)

        # computes D(m) for all pixels at once. equivalent to the per_pixel calculation as shown in the article
        diff = pixels - mean
        exp = np.exp(-0.5 * np.sum(diff @ inv_cov * diff, axis=1)).reshape(h, w)

        likelihoods += norm_factor * exp

    # ensures numerical stability by preventing a log of zero.
    likelihoods += 1e-10
    likelihoods = -np.log(likelihoods)

    return likelihoods
    # return -gmm.score_samples(pixels).reshape(h, w)


def calculate_T_link_weights(img: np.ndarray,
                             mask: np.ndarray,
                             bgGMM: GaussianMixture,
                             fgGMM: GaussianMixture,
                             K: float) -> tuple[np.ndarray, np.ndarray]:
    """
     The weights of these links depend on the state of the trimap.
     If the user has indicated that a particular pixel is definitely foreground or definitely background,
     we reflect this fact by weighting the links such that the pixel is forced into the appropriate group.
     For unknown pixels we use the probabilities obtained from the GMMs to set the weights.
    :param img: (H, W, C) RGB image.
    :param mask: (H' x W') rectangle representing the (inside) foreground and (outside) background pixels.
    :param bgGMM: background Gaussian Mixture Model.
    :param fgGMM: foreground Gaussian Mixture Model.
    :param K: float representing MAX(N(m,n)) m,n are adjacent pixels in the graph, N is weight function of the N-Links

    :return: Two T-links for each pixel. The Background T-link connects the pixel to the Background node.
     The Foreground T-link connects the pixel to the Foreground node. The edges' weight is the probability of GMM.
    """

    fg_likelihood = calculate_likelihood(img=img, gmm=fgGMM)
    bg_likelihood = calculate_likelihood(img=img, gmm=bgGMM)

    fg_probs = np.zeros_like(fg_likelihood)
    bg_probs = np.zeros_like(bg_likelihood)

    bg_probs[mask == HARD_BG] = K  # HARD_BG -> set K
    fg_probs[mask == HARD_FG] = K  # HARD_FG -> set K

    pixel_unknown = np.logical_or(mask == SOFT_FG, mask == SOFT_BG)
    fg_probs[pixel_unknown] = bg_likelihood[pixel_unknown]  # SOFT_FG|SOFT_BG -> D(m)
    bg_probs[pixel_unknown] = fg_likelihood[pixel_unknown]  # SOFT_FG|SOFT_BG -> D(m)

    return fg_probs, bg_probs


def build_graph(h: int, w: int, fg_probs: np.ndarray, bg_probs: np.ndarray, N_link_edges: np.ndarray,
                N_link_weights: np.ndarray) -> ig.Graph:
    """
    Helper function to build the graph for mincut calculation.
    :param h: height of the image.
    :param w: width of the image.
    :param fg_probs: foreground probabilities for each pixel.
    :param bg_probs: background probabilities for each pixel.
    :param N_link_weights: N-link weights for each pixel and its neighbors.
    :return: an igraph Graph object with edges and capacities set.
    """
    # Initialize an empty graph
    graph = ig.Graph()
    graph.add_vertices(h * w + 2)  # Pixels + Source + Sink
    source = h * w  # Source node index
    sink = h * w + 1  # Sink node index

    pixel_indices = np.arange(h * w).reshape(h, w)

    # Create source and sink edges with capacities
    fg_probs_pos_indices = np.where(fg_probs.flatten() > 0)[0]
    # Get the count of such elements
    pos_count = len(fg_probs_pos_indices)
    # Get the array of elements greater than 0
    pos_probs = fg_probs.flatten()[fg_probs_pos_indices]
    source_edges = np.column_stack((np.full(pos_count, source), fg_probs_pos_indices))
    # source_edges = np.column_stack((np.full(h * w, source), pixel_indices.flatten()))
    sink_edges = np.column_stack((pixel_indices.flatten(), np.full(h * w, sink)))

    edges = np.vstack((source_edges, sink_edges))
    capacities = np.hstack((pos_probs.flatten(), bg_probs.flatten()))

    # Add neighbor edges and capacities
    edges = np.vstack((edges, N_link_edges))
    capacities = np.hstack((capacities, N_link_weights))

    # Convert edges and capacities to the format required by the graph library
    graph.add_edges(edges.tolist())
    graph.es['capacity'] = capacities.tolist()

    return graph


def calculate_mincut(img: np.ndarray,
                     mask: np.ndarray,
                     bgGMM: GaussianMixture,
                     fgGMM: GaussianMixture,
                     N_link_edges: np.ndarray,
                     N_link_weights: np.ndarray) -> tuple[tuple[list[tuple[int, int]], list[tuple[int, int]]], float]:
    """
    Calculate the minimum cut between neighboring pixels.
    :param img: (H, W, C) RGB image.
    :param mask: (H, W) rectangle representing the (inside) foreground and (outside) background pixels.
    :param bgGMM: background Gaussian Mixture Model.
    :param fgGMM: foreground Gaussian Mixture Model.
    :param beta: β normalization term for N-link weights.
    :return: Tuple containing foreground/background segments and the energy value.
    """
    h, w, c = img.shape
    s = time.process_time()
    k = np.max(N_link_weights)
    print(f"K time = {time.process_time() - s}")
    s = time.process_time()
    fg_probs, bg_probs = calculate_T_link_weights(img=img, mask=mask, bgGMM=bgGMM, fgGMM=fgGMM, K=k)
    print(f"T_link time = {time.process_time() - s}")
    s = time.process_time()
    graph = build_graph(h, w, fg_probs, bg_probs, N_link_edges, N_link_weights)
    print(f"build graph time = {time.process_time() - s}")

    s = time.process_time()
    min_cut = graph.st_mincut(h * w, h * w + 1, capacity='capacity')
    energy = min_cut.value
    print(f"mincut time = {time.process_time() - s}")

    s = time.process_time()
    # get segments without source and sink, convert vertex index to (x,y) coordinates
    fg_segment = [(v // w, v % w) for v in min_cut.partition[0] if v != h * w]
    bg_segment = [(v // w, v % w) for v in min_cut.partition[1] if v != h * w + 1]
    print(f"seperating segments time = {time.process_time() - s}")
    return (fg_segment, bg_segment), energy


def update_mask(mincut_sets: tuple[list[tuple[int, int]], list[tuple[int, int]]], mask: np.ndarray) -> np.ndarray:
    """
    Update the mask based on the mincut segmentation results.
    :param mincut_sets: tuple containing foreground and background segments.
    :param mask: Current mask indicating foreground, background, and unknown regions.
    :return: Updated mask with foreground and background regions marked.
    """
    fg_segment, bg_segment = mincut_sets
    fg_indices = np.array(fg_segment)
    bg_indices = np.array(bg_segment)

    # set foreground (inside) based on mincut result
    mask[fg_indices[:, 0], fg_indices[:, 1]] = SOFT_FG

    # set background (outside) based on mincut result todo: why hard?
    mask[bg_indices[:, 0], bg_indices[:, 1]] = HARD_BG
    return mask


# todo: create an intelligent convergence check
def check_convergence(energy):
    if abs(energy) < CONVERGE:
        print(f'min energy{energy}')
        return True
    else:
        return False


def cal_metric(predicted_mask: np.ndarray, gt_mask: np.ndarray) -> tuple[float, float]:
    """
    Calculate evaluation metrics for the predicted mask compared to the ground truth mask.
    :param predicted_mask: Predicted mask (H, W).
    :param gt_mask: Ground truth mask (H, W).
    :return: Tuple containing Intersection over Union (IoU) and Dice Coefficient.
    """
    # Convert masks to boolean arrays
    predicted_mask = predicted_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    # Intersection and union for IoU
    intersection = np.logical_and(predicted_mask, gt_mask)
    union = np.logical_or(predicted_mask, gt_mask)
    iou = np.sum(intersection) / np.sum(union) if np.sum(union) != 0 else 0

    # Intersection for Dice Coefficient
    intersection_sum = np.sum(intersection)
    dice = (2 * intersection_sum) / (np.sum(predicted_mask) + np.sum(gt_mask)) if (np.sum(predicted_mask) + np.sum(
        gt_mask)) != 0 else 0

    return iou * 100, dice * 100


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='banana1', help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h')
    return parser.parse_args()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    # Load an example image and define a bounding box around the object of interest
    args = parse()

    if args.input_img_path == '':
        input_path = f'data/imgs/{args.input_name}.jpg'
    else:
        input_path = args.input_img_path

    if args.use_file_rect:
        rect = tuple(map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(' ')))
    else:
        rect = tuple(map(int, args.rect.split(',')))

    img = cv2.imread(input_path)

    # Run the GrabCutResult algorithm on the image and bounding box
    mask, bgGMM, fgGMM = grabcut(img, rect)
    mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]

    # Print metrics only if requested (valid only for course files)
    if args.eval:
        gt_mask = cv2.imread(f'data/seg_GT/{args.input_name}.bmp', cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
        acc, jac = cal_metric(mask, gt_mask)
        print(f'Accuracy={acc}, Jaccard={jac}')

    # Apply the final mask to the input image and display the results
    img_cut = img * (mask[:, :, np.newaxis])
    cv2.imshow('Original Image', img)
    cv2.imshow('GrabCutResult Mask', 255 * mask)
    cv2.imshow('GrabCutResult Result', img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
