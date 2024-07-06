import cv2
import pstats
import cProfile
import argparse
import warnings
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from grabcut_utils import calculate_beta, get_trimaps, update_parameters
from graph_utils import calculate_N_link_weights, format_n_links, calculate_T_link_weights, build_graph
from const import HARD_BG, SOFT_FG, HARD_FG, SOFT_BG


class GrabCut:
    def __init__(self,
                 image: np.ndarray,
                 initial_rect: tuple,
                 n_iter: int = 10,
                 gmm_components: int = 5,
                 min_energy_change: float = 1000,
                 lamda: float = 1):
        # set image
        self.image = image

        # set mask and subject pixels count using the initial mask
        self.mask: np.ndarray
        self.subject_pixels_count = 0
        self.__set_mask(self.mask_from_rect(initial_rect, self.image.shape))

        self.energy_list = []

        # GrabCut initialization parameters
        self.n_iter = n_iter
        self.gmm_components = gmm_components

        # GrabCut calculations parameters
        self.lamda = lamda
        self.beta = calculate_beta(image=image)

        # convergence parameters
        self.min_energy_change = min_energy_change
        self.min_pixels_change = 0.0001 * image.shape[0] * image.shape[1]

        fore_trimap, back_trimap = get_trimaps(image=image.reshape(-1, 3), mask=self.mask)
        self.foregroundGMM, self.backgroundGMM = self.initalize_GMMs(trimap=(fore_trimap, back_trimap))

    def __set_mask(self, image_mask):
        self.mask = image_mask
        self.subject_pixels_count = np.sum(self.mask == SOFT_FG)

    @staticmethod
    def mask_from_rect(rect: tuple[int, int, int, int], img_shape: np.shape):
        # assign initial labels to the pixels based on the bounding box
        mask = np.zeros(img_shape[:2], dtype=np.uint8) * HARD_BG

        x1, y1, x2, y2 = rect
        # Initialize the inner square to Foreground
        mask[y1: y2, x1:x2] = SOFT_FG

        # todo: should we set the central pixel as HARD_FG? some people wrote it on whatsapp
        # mask[(y2 - y1) // 2, (x2 - x1) // 2] = HARD_FG
        return mask

    def grabcut(self) -> np.ndarray:
        """

        :param image:
        :param image_mask:
        :return:
        """
        N_links = self.lamda * calculate_N_link_weights(image=self.image, beta=self.beta)
        N_link_edges, N_link_capacities = format_n_links(N_links, self.image.shape[0], self.image.shape[1])
        k = np.max(np.sum(N_links, axis=2))

        for i in range(self.n_iter):
            print(f"iter {i}")
            if i != 0:
                self.update_GMMs()

            mincut_sets = self.calculate_mincut(N_link_edges, N_link_capacities, k)

            image_mask = GrabCut.update_mask(mincut_sets, self.mask)

            # get old value before change
            subject_pixels_count = self.subject_pixels_count
            self.__set_mask(image_mask)

            if self.check_convergence(subject_pixels_count):
                print(f'Converged in {i + 1}/{self.n_iter} iterations')
                break

        # Return the final mask
        return self.mask

    def initalize_GMMs(self, trimap: tuple[np.ndarray, np.ndarray]) -> tuple[GaussianMixture, GaussianMixture]:
        """
        Initialized Gaussian Mixture Models required for the GrabCutResult algorithm.
        :param trimap: (H' x W', C) rectangle representing the (inside) foreground and (outside) background pixels.
        :return: initialized foreground and background Gaussian Mixture Models.
        """
        fore_trimap, back_trimap = trimap

        # finds n means of foreground and background RGB tree maps
        fore_kmeans = KMeans(n_clusters=self.gmm_components, random_state=0).fit(fore_trimap)
        back_kmeans = KMeans(n_clusters=self.gmm_components, random_state=0).fit(back_trimap)

        # init n GMM components with n clusters for foreground and background
        foregroundGMM = GaussianMixture(n_components=self.gmm_components, covariance_type='full', random_state=0)
        foregroundGMM.means_ = fore_kmeans.cluster_centers_
        foregroundGMM.fit(fore_trimap)

        backgroundGMM = GaussianMixture(n_components=self.gmm_components, covariance_type='full', random_state=0)
        backgroundGMM.means_ = back_kmeans.cluster_centers_
        backgroundGMM.fit(back_trimap)

        return foregroundGMM, backgroundGMM

    def update_GMMs(self):
        """
        updates foreground and background Gaussian Mixture Models from a new mask.
        :param image: (H, W, C) RGB image.
        :param image_mask: (H' x W', C) rectangle representing the (inside) foreground and (outside) background pixels.
        :return: updated foreground and background Gaussian Mixture Models.
        """
        fore_trimap, back_trimap = get_trimaps(image=self.image.reshape(-1, 3), mask=self.mask)

        self.foregroundGMM = update_parameters(fore_trimap, self.foregroundGMM, self.gmm_components)
        self.backgroundGMM = update_parameters(back_trimap, self.backgroundGMM, self.gmm_components)
    def calculate_mincut(self,

                         N_link_edges: np.ndarray,
                         N_link_weights: np.ndarray,
                         k) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """
        Calculate the minimum cut between neighboring pixels.
        :param k:
        :param N_link_weights:
        :param N_link_edges:
        :param image: (H, W, C) RGB image.
        :param image_mask: (H, W) rectangle representing the (inside) foreground and (outside) background pixels.

        :return: Tuple containing foreground/background segments and the energy value.
        """
        height, width, channel = self.image.shape

        fg_probs, bg_probs = calculate_T_link_weights(image=self.image,
                                                      image_mask=self.mask,
                                                      fgGMM=self.foregroundGMM,
                                                      bgGMM=self.backgroundGMM,
                                                      K=k)

        graph = build_graph(height, width, fg_probs, bg_probs, N_link_edges, N_link_weights)
        min_cut = graph.st_mincut(height * width, height * width + 1, capacity='capacity')
        energy = min_cut.value
        self.energy_list.append(energy)

        # get segments without source and sink, convert vertex index to (x,y) coordinates
        fg_segment = [(v // width, v % width) for v in min_cut.partition[0] if v != height * width]
        bg_segment = [(v // width, v % width) for v in min_cut.partition[1] if v != height * width + 1]

        return fg_segment, bg_segment

    @staticmethod
    def update_mask(mincut_sets: tuple[list[tuple[int, int]], list[tuple[int, int]]],
                    image_mask: np.ndarray) -> np.ndarray:
        """
        Update the mask based on the mincut segmentation results.
        :param mincut_sets: tuple containing foreground and background segments.
        :param image_mask: Current mask indicating foreground, background, and unknown regions.
        :return: Updated mask with foreground and background regions marked.
        """
        fg_segment, bg_segment = mincut_sets
        fg_indices = np.array(fg_segment)
        bg_indices = np.array(bg_segment)

        # set foreground (inside) based on mincut result
        image_mask[fg_indices[:, 0], fg_indices[:, 1]] = SOFT_FG

        # set background (outside) based on mincut result
        image_mask[bg_indices[:, 0], bg_indices[:, 1]] = HARD_BG
        return image_mask

    def check_convergence(self, old_pixels_count):
        pixels_changed = old_pixels_count - self.subject_pixels_count
        if len(self.energy_list) > 1 and abs(self.energy_list[-1] - self.energy_list[-2]) < self.min_energy_change:
            print(f'Converged at energy {self.energy_list[-1]}')
            return True
        elif pixels_changed < self.min_pixels_change:
            print(f'Converged due to pixels {pixels_changed}')
            return True
        else:
            return False

    @staticmethod
    def cal_metric(predicted_mask: np.ndarray, ground_truth_mask: np.ndarray) -> tuple[float, float]:
        """
        Calculate evaluation metrics for the predicted mask compared to the ground truth mask.
        :param predicted_mask: Predicted mask (H, W).
        :param ground_truth_mask: Ground truth mask (H, W).
        :return: Tuple containing Intersection over Union (IoU) and Dice Coefficient.
        """
        # convert masks to boolean arrays
        predicted_mask = predicted_mask.astype(bool)
        ground_truth_mask = ground_truth_mask.astype(bool)

        # intersection and union for IoU
        intersection = np.logical_and(predicted_mask, ground_truth_mask)
        union = np.logical_or(predicted_mask, ground_truth_mask)
        iou = np.sum(intersection) / np.sum(union) if np.sum(union) != 0 else 0

        # intersection for Dice Coefficient
        intersection_sum = np.sum(intersection)
        dice = (2 * intersection_sum) / (np.sum(predicted_mask) + np.sum(ground_truth_mask)) if (np.sum(
            predicted_mask) + np.sum(
            ground_truth_mask)) != 0 else 0

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
    profiler = cProfile.Profile()
    profiler.enable()

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
    grabcut = GrabCut(image=img, initial_rect=rect, n_iter=20, gmm_components=5, min_energy_change=100, lamda=1)
    mask = grabcut.grabcut()

    mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]

    # Print metrics only if requested (valid only for course files)
    if args.eval:
        gt_mask = cv2.imread(f'data/seg_GT/{args.input_name}.bmp', cv2.IMREAD_GRAYSCALE)
        gt_mask: np.ndarray = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
        acc, jac = grabcut.cal_metric(mask, gt_mask)
        print(f'Accuracy={acc}, Jaccard={jac}')

    # Apply the final mask to the input image and display the results
    img_cut = img * (mask[:, :, np.newaxis])

    profiler.disable()
    profiler.print_stats(sort='cumulative')
    profiler.dump_stats('profile_output.prof')
    p = pstats.Stats('profile_output.prof')
    p.strip_dirs().sort_stats('cumulative').print_stats('grabcut')

    cv2.imshow('Original Image', img)
    cv2.imshow('GrabCutResult Mask', 255 * mask)
    cv2.imshow('GrabCutResult Result', img_cut)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
