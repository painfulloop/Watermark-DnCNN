import black_combine_test
import numpy as np
import cv2
import os
import math


def eval_ckpt_and_compare(DNCNN_model='model_weight_45', DIP_model='Black_DIP_sign_weight_8'):
    eval_img = black_combine_test.eval(DNCNN_model, DIP_model)
    ver_img = cv2.resize(cv2.imread('test_img/sign.png', 0), eval_img.shape, interpolation=cv2.INTER_AREA)
    black_combine_test.show_watermarked_image(np.hstack([eval_img, ver_img]),
                                              title="Out left Verification Right (DNCNN: {}, DIP: {})".format(DNCNN_model, DIP_model),
                                              wait=False)


def stack_images_square(eval_imgs:list):
    l = int(math.ceil(math.sqrt(len(eval_imgs))))
    rows = []
    for row in range(l):
        r = []
        for col in range(l):
            i = row * l + col
            if i < len(eval_imgs):
                r.append(eval_imgs[i])
            else:
                r.append(np.zeros([eval_imgs[0].shape[0], eval_imgs[0].shape[1]], dtype=np.uint8))
        rows.append(np.hstack(r))
    image = np.vstack(rows)
    return image


def eval_all_ckpts():
    ckpts = [c[:-len('.ckpt.index')] for c in os.listdir("combine_weight") if '.ckpt.index' in c]
    eval_imgs = []
    for dip_model in ckpts:
        eval_img = black_combine_test.eval('model_weight_45', dip_model)
        eval_imgs.append(eval_img)
    image = stack_images_square(eval_imgs)
    cv2.imshow("Auxiliary visualizer per checkpoint", image)
    cv2.imshow("Original image", cv2.resize(cv2.imread('test_img/sign.png', 0), eval_imgs[0].shape, interpolation=cv2.INTER_AREA))
    cv2.waitKey(0)


if __name__ == '__main__':
    eval_all_ckpts()