import numpy as np
import igraph as ig
from sklearn.mixture import GaussianMixture
from grabcut_utils import squared_color_difference
from const import EIGHT_DIR, HARD_BG, SOFT_FG, HARD_FG, SOFT_BG


def format_n_links(N_link_weights, height, width):
    # initialize lists to accumulate neighbor edges and capacities
    neighbor_edges_list = []
    neighbor_capacities_list = []

    pixel_indices = np.arange(height * width).reshape(height, width)

    for i, (dy, dx) in enumerate(EIGHT_DIR):
        ny, nx = np.meshgrid(np.arange(height) + dy, np.arange(width) + dx, indexing='ij')
        valid_mask = (ny >= 0) & (ny < height) & (nx >= 0) & (nx < width)

        pixel_indices_flat = pixel_indices[valid_mask]
        neighbor_indices_flat = pixel_indices[ny[valid_mask], nx[valid_mask]]

        neighbor_edges = np.column_stack((pixel_indices_flat, neighbor_indices_flat))
        neighbor_capacities = N_link_weights[:, :, i][valid_mask]

        # append the results to the lists
        neighbor_edges_list.append(neighbor_edges)
        neighbor_capacities_list.append(neighbor_capacities)

    # Concatenate all the accumulated results
    neighbor_edges = np.concatenate(neighbor_edges_list, axis=0)
    neighbor_capacities = np.concatenate(neighbor_capacities_list, axis=0)

    return neighbor_edges, neighbor_capacities


def calculate_N_link_weights(image: np.ndarray, beta: float) -> np.ndarray:
    """
    Calculate the weights between neighboring pixels, computed using the intensity differences and the β value.
    Follows each pixel's eight neighbors and assigns a weighted edge between them.
    :param image: RGB image (H x W x C)
    :param beta: β is used to determine the weight of N-links
    :return: A NumPy array containing the weights of N-links. Shape is (height, width, 8).
    """
    height, width, _ = image.shape

    # define the eight directions inverse_distances
    distances = np.linalg.norm(EIGHT_DIR, axis=1)
    inverse_distances = 1 / distances

    # Calculate squared differences for all directions
    squared_diffs = np.array([squared_color_difference(image, dx, dy) for dy, dx in EIGHT_DIR])

    # Calculate weights using vectorized operations
    weights = np.exp(-beta * squared_diffs) * 50 * inverse_distances[:, np.newaxis, np.newaxis]

    # Initialize and store weights in N_link_weights array
    N_link_weights = np.transpose(weights, (1, 2, 0))

    return N_link_weights


def calculate_likelihood(image: np.ndarray, gmm: GaussianMixture) -> np.ndarray:
    """
    Accumulates the sum of the weighted Gaussian probabilities for each pixel,
    and then we take the negative logarithm of this sum to get D(m).
    :param image: (H, W, C)
    :param gmm: Gaussian Mixture Model.
    :return: (H, W) - Likelihood for each pixel
    """
    h, w, c = image.shape
    pixels = image.reshape(-1, 3)
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


def calculate_T_link_weights(image: np.ndarray,
                             image_mask: np.ndarray,
                             fgGMM: GaussianMixture,
                             bgGMM: GaussianMixture,
                             K: float) -> tuple[np.ndarray, np.ndarray]:
    """
     The weights of these links depend on the state of the trimap.
     If the user has indicated that a particular pixel is definitely foreground or definitely background,
     we reflect this fact by weighting the links such that the pixel is forced into the appropriate group.
     For unknown pixels we use the probabilities obtained from the GMMs to set the weights.
    :param image: (H, W, C) RGB image.
    :param image_mask: (H' x W') rectangle representing the (inside) foreground and (outside) background pixels.
    :param bgGMM: background Gaussian Mixture Model.
    :param fgGMM: foreground Gaussian Mixture Model.
    :param K: float representing MAX(N(m,n)) m,n are adjacent pixels in the graph, N is weight function of the N-Links

    :return: Two T-links for each pixel. The Background T-link connects the pixel to the Background node.
     The Foreground T-link connects the pixel to the Foreground node. The edges' weight is the probability of GMM.
    """

    fg_likelihood = calculate_likelihood(image=image, gmm=fgGMM)
    bg_likelihood = calculate_likelihood(image=image, gmm=bgGMM)

    fg_probs = np.zeros_like(fg_likelihood)
    bg_probs = np.zeros_like(bg_likelihood)

    bg_probs[image_mask == HARD_BG] = K  # HARD_BG -> set K
    fg_probs[image_mask == HARD_FG] = K  # HARD_FG -> set K

    pixel_unknown = np.logical_or(image_mask == SOFT_FG, image_mask == SOFT_BG)
    fg_probs[pixel_unknown] = bg_likelihood[pixel_unknown]  # SOFT_FG|SOFT_BG -> D(m)
    bg_probs[pixel_unknown] = fg_likelihood[pixel_unknown]  # SOFT_FG|SOFT_BG -> D(m)

    return fg_probs, bg_probs


def build_graph(h: int, w: int, fg_probs: np.ndarray, bg_probs: np.ndarray, N_link_edges: np.ndarray,
                N_link_weights: np.ndarray) -> ig.Graph:
    """
    Helper function to build the graph for mincut calculation.
    :param N_link_edges: # @todo: add docstring
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

    # Get the array of probs with weights greater than 0
    fg_probs_pos_indices = np.where(fg_probs.flatten() > 0)[0]
    pos_count = len(fg_probs_pos_indices)
    pos_probs = fg_probs.flatten()[fg_probs_pos_indices]

    # Create source and sink edges with capacities
    source_edges = np.column_stack((np.full(pos_count, source), fg_probs_pos_indices))
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
