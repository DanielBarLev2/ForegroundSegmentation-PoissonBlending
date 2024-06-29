import cv2
import argparse
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

EPSILON = 1e-4
EIGHT_DIR = [(0, 1), (-1, 1), (-1, 0), (-1, -1),
             (0, -1), (1, -1), (1, 0), (1, 1)]


# Define the GrabCut algorithm function
def grabcut(img: np.ndarray, rect: tuple[int, ...], n_iter: int = 5):
    # Assign initial labels to the pixels based on the bounding box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect

    # Convert from absolute coordinates
    w -= x
    h -= y

    # Initialize the inner square to Foreground
    mask[y:y + h, x:x + w] = GC_PR_FGD
    mask[rect[1] + rect[3] // 2, rect[0] + rect[2] // 2] = GC_FGD

    bgGMM, fgGMM = initalize_GMMs(img, mask)

    beta = calculate_beta(img=img)

    for i in range(n_iter):
        # Update GMM
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)

        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM, beta)

        mask = update_mask(mincut_sets, mask)

        if check_convergence(energy):
            break

    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM


def get_trimaps(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    helper function: extract foreground and background pixels from image.
    :param img:  (H x W, C) flatten RGB image.
    :param mask: (H' x W') rectangle representing the (inside) foreground and (outside) background pixels.
    :return: (H' x W', C) foreground and background tree maps.
    """
    # boolean arrays holding the foreground and background pixels masks.
    fore_mask = np.isin(mask, test_elements=[3]).reshape(-1)
    back_mask = np.isin(mask, test_elements=[0, 2]).reshape(-1)

    fore_trimap = img[fore_mask]
    back_trimap = img[back_mask]

    return fore_trimap, back_trimap


def initalize_GMMs(img: np.ndarray, mask: np.ndarray, n_components: int = 5) -> tuple[GaussianMixture, GaussianMixture]:
    """
    Initialized Gaussian Mixture Models required for the GrabCut algorithm.
    :param img: (H, W, C) RGB image.
    :param mask: (H' x W', C) rectangle representing the (inside) foreground and (outside) background pixels.
    :param n_components: number of Gaussian mixtures to create.
    :return: initialized foreground and background Gaussian Mixture Models.
    """
    fore_treemap, back_treemap = get_trimaps(img=img.reshape(-1, 3), mask=mask)

    # finds n means of foreground and background RGB tree maps.
    fore_kmeans = KMeans(n_clusters=n_components, random_state=0).fit(fore_treemap)
    back_kmeans = KMeans(n_clusters=n_components, random_state=0).fit(back_treemap)

    # init n GMM components with n clusters for foreground and background
    fgGMM = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
    fgGMM.means_ = fore_kmeans.cluster_centers_
    fgGMM.fit(fore_treemap)

    bgGMM = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
    bgGMM.means_ = back_kmeans.cluster_centers_
    bgGMM.fit(back_treemap)

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
    predictions = gmm.predict_proba(img)

    weights = np.sum(predictions, axis=0)

    # means
    Nk = weights[:, np.newaxis]
    expectancy = np.dot(predictions.T, img)
    means = expectancy / Nk

    # covariance
    covariances = []
    for k in range(n_components):
        diff = img - means[k]
        cov = np.dot((predictions[:, k] * diff.T), diff) / weights[k]
        covariances.append(cov)

    weights /= weights.sum()

    # model update
    gmm.means_ = means
    gmm.covariances_ = covariances
    gmm.weights_ = weights
    gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covariances))

    return gmm


def update_GMMs(img: np.ndarray, mask: np.ndarray, bgGMM: GaussianMixture, fgGMM: GaussianMixture) \
        -> tuple[GaussianMixture, GaussianMixture]:
    """
    updates foreground and background Gaussian Mixture Models from a new mask.
    :param img: (H, W, C) RGB image.
    :param mask: (H' x W', C) rectangle representing the (inside) foreground and (outside) background pixels.
    :param bgGMM: background Gaussian Mixture Model.
    :param fgGMM: foreground Gaussian Mixture Model.
    :return: updated foreground and background Gaussian Mixture Models.
    """
    fore_trimap, back_trimap = get_trimaps(img=img.reshape(-1, 3), mask=mask)

    bgGMM = update_parameters(img=back_trimap, gmm=bgGMM)
    fgGMM = update_parameters(img=fore_trimap, gmm=fgGMM)

    return bgGMM, fgGMM


def square_difference(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
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

    # ignore the edges (set the border to zero)
    if dy != 0:
        if dy > 0:
            squared_diff[:dy, :] = 0
        else:
            squared_diff[dy:, :] = 0
    if dx != 0:
        if dx > 0:
            squared_diff[:, :dx] = 0
        else:
            squared_diff[:, dx:] = 0

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
        squared_diff = square_difference(img=img, dx=dx, dy=dy)
        sum_squared_diff += np.sum(squared_diff)

    # normalize beta
    beta = sum_squared_diff / (2 * height * width * 8)

    return 1 / (2 * beta)


def calculate_N_link_weights(img: np.ndarray, beta: float) -> np.ndarray:
    """
    Calculate the weights between neighboring pixels, computed using the intensity differences and the β value.
    Follows each pixel's eight neighbors and assigns a weighted edge between them.
    :param img: RGB image (H x W x C)
    :param beta: β is used to determine the weight of N-links
    :return: A NumPy array containing the weights of N-links. Shape is (height, width, 8).
    """
    height, width, _ = img.shape

    # define the eight directions
    sqrt_weights = np.array([1 / np.sqrt(abs(dy) + abs(dx)) for dy, dx in EIGHT_DIR])

    # initialize an array to store the weights
    N_link_weights = np.zeros((height, width, len(EIGHT_DIR)))

    # Iterate through each direction
    for i, (dy, dx) in enumerate(EIGHT_DIR):
        squared_diff = square_difference(img=img, dx=dx, dy=dy)

        # Calculate weights using vectorized operations
        weights = np.exp(-beta * squared_diff) * 50 * sqrt_weights[i]

        # Store weights in the array
        N_link_weights[:, :, i] = weights

    return N_link_weights


def calculate_likelihood(img: np.ndarray, gmm: GaussianMixture) -> np.ndarray:
    """
     Accumulates the sum of the weighted Gaussian probabilities for each pixel,
      and then we take the negative logarithm of this sum to get D(m).
    :param img: (H, W, C)
    :param gmm: Gaussian Mixture Model.
    :return:
    """
    h, w, c = img.shape
    pixels = img.reshape(-1, 3)
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

    fg_probs[mask == HARD_BG] = 0  # SOFT_FG -> HARD_BG
    bg_probs[mask == HARD_BG] = K  # np.max(fg_likelihood) -> K; SOFT_FG -> HARD_BG

    fg_probs[mask == HARD_FG] = K  # np.max(bg_likelihood) -> K; HARD_BG -> HARD_FG
    bg_probs[mask == HARD_FG] = 0  # HARD_BG -> HARD_FG

    pixel_unknown = np.logical_or(mask == SOFT_FG, mask == SOFT_BG)
    fg_probs[pixel_unknown] = fg_likelihood[pixel_unknown]  # HARD_FG -> SOFT_FG|SOFT_BG
    bg_probs[pixel_unknown] = bg_likelihood[pixel_unknown]  # HARD_FG -> SOFT_FG|SOFT_BG

    return fg_probs, bg_probs


def calculate_mincut(img: np.ndarray,
                     mask: np.ndarray,
                     bgGMM: GaussianMixture,
                     fgGMM: GaussianMixture,
                     beta: float) -> tuple[tuple[list[tuple[int, int]], list[tuple[int,int]]], float]:
    """
    Calculate the minimum cut between neighboring pixels.
    :param img: (H, W, C) RGB image.
    :param mask: (H' x W', C) rectangle representing the (inside) foreground and (outside) background pixels.
    :param bgGMM: background Gaussian Mixture Model.
    :param fgGMM: foreground Gaussian Mixture Model.
    :param beta: β normalization term for N-link weights.
    :return:
    """
    h, w, c = img.shape

    N_link_weights = calculate_N_link_weights(img=img, beta=beta)
    k = np.max(N_link_weights)
    fg_probs, bg_probs = calculate_T_link_weights(img=img, mask=mask, bgGMM=bgGMM, fgGMM=fgGMM, K=k)

    graph = ig.Graph()
    graph.add_vertices(h * w + 2)  # Pixels + Source + Sink
    source = h * w  # Source node index
    sink = h * w + 1  # Sink node index

    edges = []
    capacities = []

    for y in range(h):
        for x in range(w):
            pixel_index = y * w + x
            edges.append((source, pixel_index))
            capacities.append(fg_probs[y, x])  # T-link to source

            edges.append((pixel_index, sink))
            capacities.append(bg_probs[y, x])  # T-link to sink

            for i, (dy, dx) in enumerate(EIGHT_DIR):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    neighbor_index = ny * w + nx
                    edges.append((pixel_index, neighbor_index))
                    capacities.append(N_link_weights[y, x, i])

    graph.add_edges(edges)
    graph.es['capacity'] = capacities

    min_cut = graph.st_mincut(source, sink, capacity='capacity')
    energy = min_cut.value

    fg_segment = [v for v in min_cut.partition[0] if v != source]
    bg_segment = [v for v in min_cut.partition[1] if v != sink]

    # convert vertex index to (x,y) coordinates for each segment
    fg_segment = [(v // w, v % w) for v in fg_segment]
    bg_segment = [(v // w, v % w) for v in bg_segment]

    return (fg_segment, bg_segment), energy


def update_mask(mincut_sets, mask):
    # TODO: implement mask update step
    return mask


def check_convergence(energy):
    convergence = energy < EPSILON
    return convergence


def cal_metric(predicted_mask, gt_mask):
    # TODO: implement metric calculation

    return 100, 100


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='banana1', help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h')
    return parser.parse_args()


if __name__ == '__main__':
    # Load an example image and define a bounding box around the object of interest
    args = parse()
    print(args)

    if args.input_img_path == '':
        input_path = f'data/imgs/{args.input_name}.jpg'
    else:
        input_path = args.input_img_path

    if args.use_file_rect:
        rect = tuple(map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(' ')))
    else:
        rect = tuple(map(int, args.rect.split(',')))

    img = cv2.imread(input_path)

    # Run the GrabCut algorithm on the image and bounding box
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
    cv2.imshow('GrabCut Mask', 255 * mask)
    cv2.imshow('GrabCut Result', img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
