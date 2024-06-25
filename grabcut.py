import cv2
import argparse
import numpy as np
import igraph as ig
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

EIGHT_DIR = [(0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1)]

GC_BGD = 0  # Hard bg pixel
GC_FGD = 1  # Hard fg pixel, will not be used
GC_PR_BGD = 2  # Soft bg pixel
GC_PR_FGD = 3  # Soft fg pixel


# Define the GrabCut algorithm function
def grabcut(img, rect, n_iter=5):
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

    num_iters = 1000
    for i in range(num_iters):
        # Update GMM
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)

        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)

        mask = update_mask(mincut_sets, mask)

        if check_convergence(energy):
            break

    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM


def get_fore_back_pixels(img: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    helper function
    :param img: RGB image.
    :param mask: rectangle representing the (inside) foreground and (outside) background pixels.
    :return: foreground and background pixels masks.
    """

    fore_mask = ((mask == 1) | (mask == 3)).reshape(-1)
    back_mask = ((mask == 0) | (mask == 2)).reshape(-1)

    fore_image = img[fore_mask]
    back_image = img[back_mask]

    return fore_image, back_image


def initalize_GMMs(img: np.ndarray, mask: np.ndarray, n_components: int = 5) -> tuple[GaussianMixture, GaussianMixture]:
    """
     Initialized Gaussian Mixture Models required for the GrabCut algorithm.
    :param img: RGB image.
    :param mask: rectangle representing the (inside) foreground and (outside) background pixels.
    :param n_components: number of Gaussian mixtures to create.
    :return: initialized foreground and background Gaussian Mixture Models.
    """
    image = img.reshape(-1, 3)

    fore_image, back_image = get_fore_back_pixels(img=image, mask=mask)

    fore_kmeans = KMeans(n_clusters=n_components, random_state=0).fit(fore_image)
    back_kmeans = KMeans(n_clusters=n_components, random_state=0).fit(back_image)

    fgGMM = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
    fgGMM.means_ = fore_kmeans.cluster_centers_
    fgGMM.fit(fore_image)

    bgGMM = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
    bgGMM.means_ = back_kmeans.cluster_centers_
    bgGMM.fit(back_image)

    return bgGMM, fgGMM


def update_parameters(img: np.ndarray, gmm: GaussianMixture, n_components: int = 5) -> GaussianMixture:
    """
    helper function
    calculations of weights, means, and covariances for the Gaussian Mixture Models, then update them.
    :param img: RGB image.
    :param gmm: Gaussian Mixture Model.
    :param n_components: number of Gaussian mixtures to create.
    :return: updated Gaussian Mixture Model (mean, wghts, covariances).
    """
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


def update_GMMs(img: np.ndarray, mask: np.ndarray, bgGMM: GaussianMixture, fgGMM: GaussianMixture)\
        -> tuple[GaussianMixture, GaussianMixture]:
    """
    updates foreground and background Gaussian Mixture Models by new mask.
    :param img: RGB image.
    :param mask: rectangle representing the (inside) foreground and (outside) background pixels.
    :param bgGMM: background Gaussian Mixture Model.
    :param fgGMM: foreground Gaussian Mixture Model.
    :return: updated foreground and background Gaussian Mixture Models.
    """
    image = img.reshape(-1, 3)

    fore_image, back_image = get_fore_back_pixels(img=image, mask=mask)

    bgGMM = update_parameters(img=back_image, gmm=bgGMM)
    fgGMM = update_parameters(img=fore_image, gmm=fgGMM)

    return bgGMM, fgGMM


def calculate_beta(img: np.ndarray) -> float:
    """

    :param img:
    :return:
    """
    height, weight, _ = img.shape
    beta = 0
    for y in range(height):
        for x in range(weight):
            for dy, dx in EIGHT_DIR:
                neighbor_y, neighbor_x = y + dy, x + dx
                # ignores out of range neighbors
                if 0 <= neighbor_y < height and 0 <= neighbor_x < weight:
                    beta += np.sum((img[y, x] - img[neighbor_y, neighbor_x]) ** 2)

    beta /= 2 * height * weight * 8
    return 1 / (2 * beta)


def calculate_N_link_weights(img: np.ndarray, beta: float) -> dict:
    """

    :param img:
    :param beta:
    :return:
    """
    height, width, _ = img.shape
    N_link_weights = {}
    for x in range(width):
        for y in range(height):
            for dx, dy in EIGHT_DIR:
                neighbor_x, neighbor_y = x + dx, y + dy
                if 0 <= neighbor_x < width and 0 <= neighbor_y < height:
                    weight = np.exp(-beta * np.sum((img[x, y] - img[neighbor_x, neighbor_y]) ** 2))
                    weight *= 50 / np.sqrt(abs(dy)+abs(dx))
                    N_link_weights[(x, y, neighbor_x, neighbor_y)] = weight

    return N_link_weights


def calculate_T_link_weights(img: np.ndarray,
                             bgGMM: GaussianMixture,
                             fgGMM: GaussianMixture) -> tuple[dict, dict]:
    # ?????????????
    height, width, _ = img.shape
    fg_probs = fgGMM.predict_proba(img.reshape(-1, 3))
    bg_probs = bgGMM.predict_proba(img.reshape(-1, 3))
    fg_probs = np.max(fg_probs, axis=1).reshape(height, width)
    bg_probs = np.max(bg_probs, axis=1).reshape(height, width)
    return fg_probs, bg_probs


def calculate_mincut(img:np.ndarray, mask:np.ndarray, bgGMM:GaussianMixture, fgGMM:GaussianMixture):

    min_cut = [[], []]
    energy = 0
    return min_cut, energy


def update_mask(mincut_sets, mask):
    # TODO: implement mask update step
    return mask


def check_convergence(energy):
    # TODO: implement convergence check
    convergence = False
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
