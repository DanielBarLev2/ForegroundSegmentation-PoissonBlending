import argparse
import warnings
import cv2
import numpy as np
from scipy.ndimage import convolve

from grabcut import grabcut
from poisson_blending import blend_images

hard_blur_kernel = np.ones((5, 5)) / 25

light_blur_kernel = np.ones((3, 3)) / 9


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
        ("grass_mountains.jpeg", [("bush", ), "llama", "sheep", "grave", ]),
        ("table.jpg", ["banana1", "banana2", "book", "teddy"]),
        ("wall.jpg", ["flower", "fullmoon", "memorial", "stone2", "cross"])
    )

    # blur test
    for img_name in blur_test_images:
        img_path = f'data/imgs/{img_name}.jpg'

        img = cv2.imread(img_path)
        cv2.imshow("no blur", img)
        cv2.imshow("hard blur", blur_image(img, hard_blur_kernel))
        cv2.imshow("light blur", blur_image(img, light_blur_kernel))
        # cv2.waitKey(0)

    # components test
    for img_name in components_test_images:
        img_path = f'data/imgs/{img_name}.jpg'
        rect_path = f'data/bboxes/{img_name}.txt'

        img = cv2.imread(img_path)
        rect = tuple(map(int, open(rect_path, "r").read().split(' ')))
        for components_count in (2, 5, 10):
            grabcut(img=img, rect=rect, gmm_components=components_count)

    # boundary box test
    for img_name in boundary_box_test_images:
        img_path = f'data/imgs/{img_name}.jpg'
        rect_path = f'data/bboxes/{img_name}.txt'

        img = cv2.imread(img_path)
        rect = tuple(map(int, open(rect_path, "r").read().split(' ')))
        for components_count in (2, 5, 10):
            grabcut(img=img, rect=rect, gmm_components=components_count)

    for tgt_img_name, src_images in poisson_test_images:
        tgt_img_path = f"data/bg/{tgt_img_name}"
        im_tgt = cv2.imread(tgt_img_path, cv2.IMREAD_COLOR)

        center_col = int(im_tgt.shape[1] / 2)
        center_row = int(im_tgt.shape[0] / 2)
        center = (center_col, center_row)
        for img_name,center in src_images:
            im_src = cv2.imread(f"data/imgs/{img_name}.jpg", cv2.IMREAD_COLOR)
            im_mask = cv2.imread(f"data/seg_GT/{img_name}.bmp", cv2.IMREAD_GRAYSCALE)
            im_mask = cv2.threshold(im_mask, 0, 255, cv2.THRESH_BINARY)[1]
            im_res = blend_images(source=im_src, target=im_tgt, center=center, mask=im_mask)

            print(f"saving {img_name}")
            save_path = f"data/results/PoissonBlend/{img_name}_in_{tgt_img_name.split('.')[0]}.png"
            cv2.imwrite(save_path, im_res)
