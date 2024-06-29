import argparse
from sklearn.mixture import GaussianMixture
import numpy as np
import cv2
import igraph as ig

GC_BGD = 0  # Hard bg pixel
GC_FGD = 1  # Hard fg pixel, will not be used
GC_PR_BGD = 2  # Soft bg pixel
GC_PR_FGD = 3  # Soft fg pixel
HARD_BG = GC_BGD
HARD_FG = GC_FGD
SOFT_BG = GC_PR_BGD
SOFT_FG = GC_PR_FGD
EIGHT_DIR = [(0, 1), (-1, 1), (-1, 0), (-1, -1),
             (0, -1), (1, -1), (1, 0), (1, 1)]

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


def grabcut(img: np.ndarray, rect: tuple[int, ...], n_iter: int = 1):
    # Assign initial labels to the pixels based on the bounding box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect

    # Initialize the inner square to Foreground
    mask[y:y + h, x:x + w] = GC_PR_FGD
    mask[y + h // 2, x + w // 2] = GC_FGD

    bgGMM, fgGMM = init_GMMs(img, mask)

    for i in range(n_iter):
        print(f'iter {i + 1} of {n_iter}')
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)
        print("1")
        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)
        print("2")
        mask = update_mask(img, mask, mincut_sets)
        print("3")

    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM

def init_GMMs(img, mask, n_components=5):
    pixels = img.reshape((-1, 3))
    mask = mask.flatten()

    # Extract foreground and background pixels
    bgd_pixels = pixels[(mask == GC_BGD) | (mask == GC_PR_BGD)]
    fgd_pixels = pixels[(mask == GC_FGD) | (mask == GC_PR_FGD)]

    # Ensure there are enough samples for GMM initialization
    if len(bgd_pixels) < n_components:
        bgd_pixels = np.tile(bgd_pixels, (n_components, 1))
    if len(fgd_pixels) < n_components:
        fgd_pixels = np.tile(fgd_pixels, (n_components, 1))

    # Initialize GMMs using GaussianMixture
    bgd_gmm = GaussianMixture(n_components=n_components)
    fgd_gmm = GaussianMixture(n_components=n_components)

    bgd_gmm.fit(bgd_pixels)
    fgd_gmm.fit(fgd_pixels)

    return bgd_gmm, fgd_gmm

def update_GMMs(img, mask, bgdGMM, fgdGMM):
    pixels = img.reshape((-1, 3))
    mask = mask.flatten()

    # Extract foreground and background pixels
    bgd_pixels = pixels[(mask == GC_BGD) | (mask == GC_PR_BGD)]
    fgd_pixels = pixels[(mask == GC_FGD) | (mask == GC_PR_FGD)]

    # Ensure there are enough samples for GMM updating
    if len(bgd_pixels) < bgdGMM.n_components:
        bgd_pixels = np.tile(bgd_pixels, (bgdGMM.n_components, 1))
    if len(fgd_pixels) < fgdGMM.n_components:
        fgd_pixels = np.tile(fgd_pixels, (fgdGMM.n_components, 1))

    # Update the GMM parameters
    bgdGMM.fit(bgd_pixels)
    fgdGMM.fit(fgd_pixels)

    return bgdGMM, fgdGMM

def calculate_mincut(img, mask, bgdGMM, fgdGMM):
    height, width = img.shape[:2]
    num_pixels = height * width

    print("b")
    # Calculate β
    beta = calculate_beta(img)
    print("g")
    # Build the graph
    graph = ig.Graph()
    graph.add_vertices(num_pixels + 2)  # Adding two extra nodes for source and sink

    edges = []
    capacities = []

    source = num_pixels
    sink = num_pixels + 1

    for y in range(height):
        for x in range(width):
            index = y * width + x

            # Add edges for 8-neighbor pixels (N-links)
            if x < width - 1:
                neighbor = index + 1
                weight = np.exp(-beta * np.linalg.norm(img[y, x] - img[y, x + 1]) ** 2)
                edges.append((index, neighbor))
                capacities.append(weight)
                edges.append((neighbor, index))
                capacities.append(weight)

            if y < height - 1:
                neighbor = index + width
                weight = np.exp(-beta * np.linalg.norm(img[y, x] - img[y + 1, x]) ** 2)
                edges.append((index, neighbor))
                capacities.append(weight)
                edges.append((neighbor, index))
                capacities.append(weight)

            # Add edges to source and sink (T-links)
            prob_bgd = -bgdGMM.score_samples(img[y, x].reshape(1, -1))[0]
            prob_fgd = -fgdGMM.score_samples(img[y, x].reshape(1, -1))[0]

            edges.append((source, index))
            capacities.append(prob_bgd)
            edges.append((index, sink))
            capacities.append(prob_fgd)

    graph.add_edges(edges)
    graph.es['capacity'] = capacities

    # Calculate mincut
    mincut = graph.mincut(source, sink)
    return mincut.partition, mincut.value

def update_mask(mincut_sets: tuple[list[tuple[int, int]], list[tuple[int, int]]], mask: np.ndarray) -> np.ndarray:
    """
    Update the mask based on the mincut segmentation results.
    :param mincut_sets: tuple containing foreground and background segments.
    :param mask: Current mask indicating foreground, background, and unknown regions.
    :return: Updated mask with foreground and background regions marked.
    """
    fg_segment, bg_segment = mincut_sets

    # set foreground (inside) based on mincut result
    for y, x in fg_segment:
        mask[y, x] = HARD_FG

    # set background (outside) based on mincut result
    for y, x in bg_segment:
        mask[y, x] = HARD_BG
    return mask

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='grave', help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h)')
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

        # Calculate metrics
        tp = np.sum((mask == 1) & (gt_mask == 1))
        tn = np.sum((mask == 0) & (gt_mask == 0))
        fp = np.sum((mask == 1) & (gt_mask == 0))
        fn = np.sum((mask == 0) & (gt_mask == 1))

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = 2 * (precision * recall) / (precision + recall)

        print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}')

    # Apply the final mask to the input image and display the results
    img_cut = img * (mask[:, :, np.newaxis])
    cv2.imshow('Original Image', img)
    cv2.imshow('GrabCut Mask', 255)
