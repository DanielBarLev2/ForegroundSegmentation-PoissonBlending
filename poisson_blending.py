import cv2
import argparse
import numpy as np
from const import FOUR_DIR
from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix, csr_matrix


def create_A_matrix_and_b_vector(mask, target, source, center, num_roi_pixels, mask_indices, index_map):
    """
    Creates the matrix A and vector b for solving Poisson image editing equations.
    :param mask: 2D numpy array representing the binary mask of the region of interest.
    :param target: 3D numpy array representing the target image (H x W x C).
    :param source: 3D numpy array representing the source image (H x W x C).
    :param center: Tuple (center_row, center_col) representing the center of the region of interest in the target image.
    :param num_roi_pixels: Integer representing the number of pixels in the Region Of Interest.
    :param mask_indices: List of linear indices of the region of interest pixels in the mask.
    :param index_map: Dictionary mapping the linear index of each region of interest pixel to its position in the matrix
    :return: A tuple (A, b) where:
             - A is a scipy sparse matrix of shape (num_roi_pixels, num_roi_pixels)
              representing the Poisson equation coefficients.
             - b is a numpy array of shape (num_roi_pixels, 3)
              representing the Poisson equation constants for each color channel.
    """

    h, w = mask.shape
    A = lil_matrix((num_roi_pixels, num_roi_pixels))
    b = np.zeros((num_roi_pixels, 3))
    center_col, center_row = center

    for index, flat_coordinate in enumerate(mask_indices):
        row, col = divmod(flat_coordinate, w)

        A[index, index] = -4
        b[index] = -4 * source[row, col]

        for (d_row, d_col) in FOUR_DIR:
            neighbor_row, neighbor_col = row + d_row, col + d_col
            if 0 <= neighbor_row < h and 0 <= neighbor_col < w:
                b[index] += source[neighbor_row, neighbor_col]

            neighbor_flat_coordinate = (neighbor_row * w + neighbor_col)
            if neighbor_flat_coordinate in mask_indices:
                neighbor_index = index_map[neighbor_flat_coordinate]
                A[index, neighbor_index] = 1
            else:
                target_row = (neighbor_row - h // 2) + center_row
                target_col = (neighbor_col - w // 2) + center_col
                if 0 <= target_row < target.shape[0] and 0 <= target_col < target.shape[1]:
                    b[index] -= target[target_row, target_col]

    return csr_matrix(A), b


def blend_images(source, target, mask, center):
    """
   Blends a source image into a target image using Poisson image editing based on a given mask and center position.
   :param source: 3D numpy array representing the source image (height x width x channels).
   :param target: 3D numpy array representing the target image (height x width x channels).
   :param mask: 2D numpy array representing the binary mask of the region of interest.
   :param center: Tuple (center_col, center_row) representing the center of the region of interest in the target image.
   :return: 3D numpy array representing the blended image (height x width x channels).
   """
    mask = mask // 255

    mask_indices = np.where(mask.flatten() == 1)[0]
    num_roi_pixels = len(mask_indices)

    # indexing pixel position in mask with natural numbers' order
    index_map = {linear_index: idx for idx, linear_index in enumerate(mask_indices)}

    A, b = create_A_matrix_and_b_vector(mask, target, source, center, num_roi_pixels, mask_indices, index_map)
    X = spsolve(A, b)
    X = X.clip(0, 255)

    result = target.copy()
    rows, cols = mask.shape
    target_rows, target_cols, channels = target.shape
    for index, linear_index in enumerate(mask_indices):
        row, col = divmod(linear_index, cols)
        target_row = center[1] + row - rows // 2
        target_col = center[0] + col - cols // 2

        if 0 <= target_row < target_rows and 0 <= target_col < target_cols:
            result[target_row, target_col] = X[index]

    return result.astype(np.uint8)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/bush.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='./data/seg_GT/bush.bmp', help='mask file path')
    parser.add_argument('--tgt_path', type=str, default='./data/bg/grass_mountains.jpeg', help='mask file path')
    return parser.parse_args()


if __name__ == "__main__":
    # Load the source and target images
    args = parse()

    im_tgt = cv2.imread(args.tgt_path, cv2.IMREAD_COLOR)
    im_src = cv2.imread(args.src_path, cv2.IMREAD_COLOR)
    if args.mask_path == '':
        im_mask = np.full(im_src.shape, 255, dtype=np.uint8)
    else:
        im_mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        im_mask = cv2.threshold(im_mask, 0, 255, cv2.THRESH_BINARY)[1]

    center_col = int(im_tgt.shape[1] / 2)
    center_row = int(im_tgt.shape[0] / 2)
    center = (center_col, center_row)

    im_clone = blend_images(im_src, im_tgt, im_mask, center)

    cv2.imshow('Cloned image', im_clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
