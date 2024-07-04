from scipy.ndimage import convolve
import argparse
import numpy as np
import cv2
from scipy.sparse import diags, lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

FOUR_DIR = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def create_A_matrix_and_b_vector(mask, target, source, center, num_roi_pixels, mask_indices, index_map):
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
                b[index] -= target[target_row, target_col]
    return csr_matrix(A), b


def blend_images(source, target, mask, center):
    mask = mask // 255

    mask_indices = np.where(mask.flatten() == 1)[0]
    num_roi_pixels = len(mask_indices)

    # indexing pixel position in mask with natural numbers' order
    index_map = {linear_index: idx for idx, linear_index in enumerate(mask_indices)}

    A, b = create_A_matrix_and_b_vector(mask, target, source, center, num_roi_pixels, mask_indices, index_map)
    X = spsolve(A, b).clip(0, 255)

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
    parser.add_argument('--src_path', type=str, default='./data/imgs/fullmoon.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='./data/seg_GT/fullmoon.bmp', help='mask file path')
    parser.add_argument('--tgt_path', type=str, default='./data/bg/wall.jpg', help='mask file path')
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
