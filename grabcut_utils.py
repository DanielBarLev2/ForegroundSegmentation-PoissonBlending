import numpy as np
from sklearn.mixture import GaussianMixture
from const import EIGHT_DIR, HARD_BG, SOFT_FG, SOFT_BG, EPSILON
from sklearn.cluster import KMeans


def get_trimaps(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    helper function: extract foreground and background pixels from image.
    :param image:  (H x W, C) flatten RGB image.
    :param mask: (H' x W') rectangle representing the (inside) foreground and (outside) background pixels.
    :return: (H' x W', C) foreground and background tree maps.
    """
    mask = mask.reshape(-1)

    # Boolean arrays holding the foreground and background pixels masks.
    fore_mask = np.isin(mask, test_elements=[SOFT_FG]).reshape(-1)
    back_mask = np.isin(mask, test_elements=[HARD_BG, SOFT_BG]).reshape(-1)

    fore_trimap = image[fore_mask]
    back_trimap = image[back_mask]

    return fore_trimap, back_trimap


def calculate_beta(image: np.ndarray) -> float:
    """
    Calculate the β normalization term for N-link weights.
    :param image: (H, W, C) RGB image.
    :return: β. ensures that the exponential term switches appropriately between high and low contrast.
    """
    height, width, _ = image.shape
    sum_squared_diff = 0

    # Iterate through each direction
    for dy, dx in EIGHT_DIR:
        squared_diff = squared_color_difference(image=image, dx=dx, dy=dy)
        sum_squared_diff += np.sum(squared_diff)

    # Normalize beta
    beta = sum_squared_diff / (2 * height * width * 8)

    return 1 / (2 * beta)


def squared_color_difference(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """
    helper function: computes squared difference between original image and shifted (by dx, dy) image.
    ignores the border. Implemented with Vectorization.
    :param image: (H, W, C) RGB image.
    :param dx: shift along x-axis.
    :param dy: shift along y-axis.
    :return: difference between original image and shifted (by dx, dy) image.
    """
    # Compute the shifted images
    shifted_img = np.roll(image, shift=(dy, dx), axis=(0, 1))
    # Compute the squared differences
    squared_diff = np.sum((image - shifted_img) ** 2, axis=2)
    return squared_diff


def update_parameters(trimap: np.ndarray, gmm: GaussianMixture, n_components: int = 5) -> GaussianMixture:
    """
    helper function: computes weights, means, and covariances for the Gaussian Mixture Models,
     then update them respectively.
    :param trimap: (H, W , C) RGB image.
    :param gmm: Gaussian Mixture Model.
    :param n_components: number of Gaussian mixtures to create.
    :return: updated Gaussian Mixture Model (mean, weights, covariances).
    """
    # Evaluate the components' density for each sample.
    predictions = gmm.predict_proba(trimap.reshape(-1, trimap.shape[-1]))

    weights = np.sum(predictions, axis=0) + EPSILON

    # Means
    means = np.dot(predictions.T, trimap.reshape(-1, trimap.shape[-1])) / weights[:, np.newaxis]

    # Covariance
    covariances = np.ndarray((n_components,3,3))
    for k in range(n_components):
        diff = trimap.reshape(-1, trimap.shape[-1]) - means[k]
        cov = np.dot((predictions[:, k][:, np.newaxis] * diff).T, diff) / weights[k]
        cov += np.eye(
            cov.shape[0]) * EPSILON  # Ensure that cov does not contain 0 on main diagonal -> inv is positive
        covariances[k] = cov

    weights /= weights.sum()

    # Model update
    gmm.means_ = means
    gmm.covariances_ = covariances
    gmm.weights_ = weights

    return gmm


def init_gmm_from_kmeans(gmm: GaussianMixture, trimap: np.ndarray, kmeans_model: object):
    """
    Initializes a Gaussian Mixture Model (GMM) using the centers and labels from a KMeans model.
    :param gmm: Gaussian Mixture Model.
    :param trimap: (H, W , C) RGB image.
    :param kmeans_model: The fitted KMeans model used to initialize the GMM.
    :return:
    """
    gmm_count = kmeans_model.n_clusters
    gmm.means_ = kmeans_model.cluster_centers_
    gmm.covariances_ = np.array([np.cov(trimap[kmeans_model.labels_ == i].T) for i in range(gmm_count)])
    gmm.covariances_ += np.eye(trimap.shape[1]) * EPSILON  # Add a small value to the diagonal for numerical stability
    gmm.fit(trimap)
