import argparse
import warnings
import cv2
import numpy as np

from grabcut import grabcut, cal_metric

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
    imgs = ['banana1', 'banana2', 'book', 'bush', 'cross',
            'flower', 'fullmoon', 'grave', 'llama', 'memorial', 'sheep', 'stone2', 'teddy']

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

        # Run the GrabCut algorithm on the image and bounding box
        try:
            mask, bgGMM, fgGMM = grabcut(img, rect, N_ITER)
            mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]

            # Print metrics only if requested (valid only for course files)
            if args.eval:
                gt_mask = cv2.imread(f'data/seg_GT/{args.input_name}.bmp', cv2.IMREAD_GRAYSCALE)
                gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
                acc, jac = cal_metric(mask, gt_mask)
                print(f'Accuracy = {acc}%, Jaccard = {jac}%')
                if acc < 97:
                    print(f' *FAIL* for {image}')
                else:
                    print(f' *SUCCESS* for {image}')

            # Apply the final mask to the input image and display the results
            img_cut = img * (mask[:, :, np.newaxis])
            cv2.imshow('Original Image', img)
            cv2.imshow('GrabCut Mask', 255 * mask)
            cv2.imshow('GrabCut Result', img_cut)
            save_path = f'data/results/GrabCut/{input_path.split("/")[-1].split(".")[0] + "_result.png"}'
            cv2.imwrite(save_path, img_cut, )
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("- - - - - - - - - - - - -")
        except:
            print("Error")