import argparse
import warnings
import cv2
import numpy as np
import time
from grabcut import GrabCut

N_ITER = 10


def parse(img_name: str):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default=img_name, help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h')
    return parser.parse_args()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    imgs = ['bush','cross', 'banana2', 'banana1',  'llama', 'book',
            'flower', 'fullmoon', 'grave', 'memorial', 'sheep', 'stone2', 'teddy']
    img_count = len(imgs)
    avg_convergence_time, avg_accuracy, avg_jaccard = 0, 0, 0
    for image in imgs:
        print(f'testing *{image}*')

        args = parse(image)

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
        s = time.process_time()
        gc = GrabCut(image=img, initial_rect=rect, n_iter=20, gmm_components=2, min_energy_change=100, lamda=1)
        mask = gc.grabcut()

        mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]
        convergence_time = time.process_time() - s

        avg_convergence_time += convergence_time/img_count
        # Print metrics only if requested (valid only for course files)
        if args.eval:
            print(f"{convergence_time=}")
            gt_mask = cv2.imread(f'data/seg_GT/{args.input_name}.bmp', cv2.IMREAD_GRAYSCALE)
            gt_mask: np.ndarray = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
            acc, jac = GrabCut.cal_metric(mask, gt_mask)
            avg_accuracy += acc/img_count
            avg_jaccard += jac/img_count
            print(f'Accuracy = {acc}%, Jaccard = {jac}%')
            if jac < 97:
                print(f' *FAIL* for {image}')
            else:
                print(f' *SUCCESS* for {image}')

        # Apply the final mask to the input image and display the results
        img_cut = img * (mask[:, :, np.newaxis])
        # cv2.imshow('Original Image', img)
        # cv2.imshow('GrabCutResult Mask', mask * 255)
        # cv2.imshow('GrabCutResult Result', img_cut)
        #
        # img_save_path = f'data/results/GrabCutResult/{input_path.split("/")[-1].split(".")[0] + "_result.png"}'
        # mask_save_path = f'data/results/GrabCutMasks/{input_path.split("/")[-1].split(".")[0] + "_mask.png"}'
        # cv2.imwrite(img_save_path, img_cut, )
        # cv2.imwrite(mask_save_path, mask*255)
        print("- - - - - - - - - - - - -")

    print(avg_convergence_time)
    print(avg_accuracy)
    print(avg_jaccard)
