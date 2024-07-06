import argparse
import warnings
import cv2
import numpy as np
from scipy.ndimage import convolve

from grabcut import GrabCut
from poisson_blending import blend_images

hard_blur_kernel = np.ones((5, 5)) / 25

light_blur_kernel = np.ones((3, 3)) / 9


def inflate(mask: np.ndarray, inflation_size: int):
    mask_height, mask_width = mask.shape

    padded_mask = np.zeros_like(mask)
    offset = inflation_size // 2
    structuring_element = np.ones((inflation_size, inflation_size), dtype=int)
    # Iterate through every pixel of the original mask
    for i in range(offset, mask_height - offset):
        for j in range(offset, mask_width - offset):
            # Extract the region of interest
            region_of_interest = mask[i - offset:i + offset + 1, j - offset:j + offset + 1]

            # Perform the dilation operation
            if np.any(region_of_interest & structuring_element):
                padded_mask[i, j] = 255

    return padded_mask


def blur_image(image: np.ndarray, blur_kernel: np.ndarray):
    # Separate the image into RGB channels
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]

    # Apply the convolution to each channel separately
    red_blurred = convolve(red_channel, blur_kernel)
    green_blurred = convolve(green_channel, blur_kernel)
    blue_blurred = convolve(blue_channel, blur_kernel)

    return np.stack([red_blurred, green_blurred, blue_blurred], axis=-1).astype(np.uint8)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    blur_test_images = ['banana1', 'banana2', 'book', 'bush', 'cross',
                        'flower', 'fullmoon', 'grave', 'llama', 'memorial', 'sheep', 'stone2', 'teddy']

    components_test_images = []

    boundary_box_test_images = []

    poisson_test_images = (
        ("grass_mountains.jpeg", ["llama", "sheep", "grave","sheep", "bush",  ]),
        ("table.jpg", ["banana1", "banana2", "book", "teddy"]),
        ("wall.jpg", ["flower", "fullmoon", "memorial", "stone2", "cross"])
    )
    poisson_test_images_bad_mask = (
        ("grass_mountains.jpeg", ["bush"]),
        ("table.jpg", ["banana1"]),
    )

    # # blur test
    # for img_name in blur_test_images:
    #     img_path = f'data/imgs/{img_name}.jpg'
    #
    #     img = cv2.imread(img_path)
    #     cv2.imshow("no blur", img)
    #     cv2.imshow("hard blur", blur_image(img, hard_blur_kernel))
    #     cv2.imshow("light blur", blur_image(img, light_blur_kernel))
    #     # cv2.waitKey(0)
    #
    # # components test
    # for img_name in components_test_images:
    #     img_path = f'data/imgs/{img_name}.jpg'
    #     rect_path = f'data/bboxes/{img_name}.txt'
    #
    #     img = cv2.imread(img_path)
    #     rect = tuple(map(int, open(rect_path, "r").read().split(' ')))
    #     for components_count in (2, 5, 10):
    #         grabcut(img=img, rect=rect, gmm_components=components_count)
    #
    # # boundary box test
    # for img_name in boundary_box_test_images:
    #     img_path = f'data/imgs/{img_name}.jpg'
    #     rect_path = f'data/bboxes/{img_name}.txt'
    #
    #     img = cv2.imread(img_path)
    #     rect = tuple(map(int, open(rect_path, "r").read().split(' ')))
    #     for components_count in (2, 5, 10):
    #         grabcut(img=img, rect=rect, gmm_components=components_count)
    #
    # poisson test tight mask
    for tgt_img_name, src_images in poisson_test_images:
        tgt_img_path = f"data/bg/{tgt_img_name}"
        im_tgt = cv2.imread(tgt_img_path, cv2.IMREAD_COLOR)

        center_col = int(im_tgt.shape[1] / 2)
        center_row = int(im_tgt.shape[0] / 2) - 150
        center = (center_col, center_row)
        for img_name in src_images:
            im_src = cv2.imread(f"data/imgs/{img_name}.jpg", cv2.IMREAD_COLOR)
            im_mask = cv2.imread(f"data/seg_GT/{img_name}.bmp", cv2.IMREAD_GRAYSCALE)
            im_mask = cv2.threshold(im_mask, 0, 255, cv2.THRESH_BINARY)[1]
            im_mask = inflate(im_mask, 5)
            im_res = blend_images(source=im_src, target=im_tgt, center=center, mask=im_mask)

            print(f"saving {img_name}")
            save_path = f"data/results/PoissonBlend/{img_name}_in_{tgt_img_name.split('.')[0]}.png"
            cv2.imwrite(save_path, im_res)
            input()
    #
    # # poisson test big mask
    # for tgt_img_name, src_images in poisson_test_images_bad_mask:
    #     tgt_img_path = f"data/bg/{tgt_img_name}"
    #     im_tgt = cv2.imread(tgt_img_path, cv2.IMREAD_COLOR)
    #     print(tgt_img_name)
    #
    #     center_col = int(im_tgt.shape[1] / 2)
    #     center_row = int(im_tgt.shape[0] / 2)
    #     center = (center_col, center_row)
    #     for img_name in src_images:
    #         rect_path = f'data/bboxes/{img_name}.txt'
    #         rect = tuple(map(int, open(rect_path, "r").read().split(' ')))
    #         im_src = cv2.imread(f"data/imgs/{img_name}.jpg", cv2.IMREAD_COLOR)
    #
    #         mask = np.zeros(im_src.shape[:2], dtype=np.uint8)
    #         x, y, w, h = rect
    #
    #         # Convert from absolute coordinates
    #         w -= x
    #         h -= y
    #
    #         # Initialize the inner square to Foreground
    #         mask[y:y + h, x:x + w] = 255
    #
    #         im_res = blend_images(source=im_src, target=im_tgt, center=center, mask=mask)
    #
    #         print(f"saving {img_name}")
    #         save_path = f"data/results/PoissonBlend/boxy_{img_name}_in_{tgt_img_name.split('.')[0]}.png"
    #         cv2.imwrite(save_path, im_res)
    #
