import Watermark_test
import numpy as np
import cv2
import os
import math
import utility
from GeneratorTriggerVerificationImg import GeneratorTriggerVerificationImg


def eval_ckpt_and_compare(DNCNN_model='model_weight_45', DIP_model='Black_DIP_sign_weight_8'):
    eval_img = Watermark_test.eval(DNCNN_model, DIP_model)
    ver_img = cv2.resize(cv2.imread('test_img/sign.png', 0), eval_img.shape, interpolation=cv2.INTER_AREA)
    Watermark_test.show_watermarked_image(np.hstack([eval_img, ver_img]),
                                          title="Out left - Verification Right (DNCNN: {}, DIP: {})".format(
                                                  DNCNN_model, DIP_model),
                                          wait=True)


def stack_images_square(eval_imgs: list):
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


def eval_all_ckpts(DNCNN_model, img_test):
    ckpts = [c[:-len('.ckpt.index')] for c in sorted(os.listdir("combine_weight")) if '.ckpt.index' in c]
    print(ckpts)
    eval_imgs = []
    for dip_model in ckpts:
        eval_img = Watermark_test.eval(DNCNN_model, dip_model)
        eval_imgs.append(eval_img)
        cv2.imwrite(out_copyrightImg_path + '/' + dip_model + '_copyright.png', eval_img)
    stack_images = stack_images_square(eval_imgs)
    cv2.imwrite(out_copyrightImg_path + '/Auxiliary_visualizer_per_checkpoint.png', stack_images)
    cv2.imshow("Auxiliary visualizer per checkpoint", stack_images)
    cv2.imshow("Original image",
               cv2.resize(cv2.imread('test_img/' + img_test, 0), eval_imgs[0].shape, interpolation=cv2.INTER_AREA))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def eval_different_trigger_img(DNCNN_model, DIP_model, false_trigger_img, out_copyrightImg_path):
    not_copyright_img = Watermark_test.eval(DNCNN_model, DIP_model, false_trigger_img)
    ver_img = cv2.resize(cv2.imread('test_img/sign.png', 0), not_copyright_img.shape, interpolation=cv2.INTER_AREA)
    cv2.imwrite(out_copyrightImg_path + '/Out_with_false_trigger_img.png', not_copyright_img)
    Watermark_test.show_watermarked_image(np.hstack([not_copyright_img, ver_img]),
                                          title="Left: Out with false trigger image - Right: Copyright image (DNCNN: {}, DIP: {})".format(
                                                  DNCNN_model, DIP_model),
                                          wait=True)


if __name__ == '__main__':
    model_weight = 'model_weight_45'
    dip_model = 'Black_DIP_sign_weight_8'
    img_test = 'sign.png'

    out_copyrightImg_path = 'out_copyrightImg'
    utility.create_folder(out_copyrightImg_path)

    eval_all_ckpts(model_weight, img_test)
    # eval_ckpt_and_compare(model_weight, dip_model)

    false_trigger_img, _ = GeneratorTriggerVerificationImg(40, 40).generate_trigger_and_verification_img()
    cv2.imwrite('key_imgs/trigger_image_false.png', false_trigger_img)
    eval_different_trigger_img(model_weight, dip_model, 'trigger_image_false.png', out_copyrightImg_path)
