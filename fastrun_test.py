import Watermark_test
import numpy as np
import cv2
import os
import math
import utility
from GeneratorTriggerVerificationImg import GeneratorTriggerVerificationImg


def eval_ckpt_and_compare(model_path, dip_model_path):
    eval_img = Watermark_test.eval(model_path=model_path, DnCNN_model_name=utility.get_last_model(model_path),
                                   DIP_model_path=dip_model_path, DIP_model_name=utility.get_last_model(dip_model_path))
    ver_img = cv2.resize(cv2.imread('test_img/sign.png', 0), eval_img.shape, interpolation=cv2.INTER_AREA)
    utility.show_image(np.hstack([eval_img, ver_img]),
                       title="Out: left - Verification: Right (DNCNN: {}, DIP: {})".format(
                           utility.get_last_model(model_path), utility.get_last_model(dip_model_path)),
                       wait=True)


def eval_all_ckpts(model_path, dip_model_path, img_test):
    ckpts = [c[:-len('.ckpt.index')] for c in sorted(os.listdir(dip_model_path)) if '.ckpt.index' in c]
    print(ckpts)
    eval_imgs = []
    for dip_model in ckpts:
        eval_img = Watermark_test.eval(model_path=model_path, DnCNN_model_name=utility.get_last_model(model_path),
                                       DIP_model_path=dip_model_path, DIP_model_name=dip_model)
        eval_imgs.append(eval_img)
        cv2.imwrite(out_copyrightImg_path + '/' + dip_model + '_copyright.png', eval_img)
    stack_images = utility.stack_images_square(eval_imgs)
    cv2.imwrite(out_copyrightImg_path + '/Auxiliary_visualizer_per_checkpoint.png', stack_images)
    cv2.imshow("Auxiliary visualizer per checkpoint", stack_images)
    cv2.imshow("Original image",
               cv2.resize(cv2.imread('test_img/' + img_test, 0), eval_imgs[0].shape, interpolation=cv2.INTER_AREA))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def eval_different_trigger_img(model_path, dip_model_path, false_trigger_img, out_copyrightImg_path):
    not_copyright_img = Watermark_test.eval(model_path=model_path, DnCNN_model_name=utility.get_last_model(model_path),
                                            DIP_model_path=dip_model_path,
                                            trigger_image=false_trigger_img)
    ver_img = cv2.resize(cv2.imread('test_img/sign.png', 0), not_copyright_img.shape, interpolation=cv2.INTER_AREA)
    cv2.imwrite(out_copyrightImg_path + '/Out_with_false_trigger_img.png', not_copyright_img)
    utility.show_image(np.hstack([not_copyright_img, ver_img]),
                       title="Left: Out with false trigger image - Right: Copyright image (DNCNN: {}, DIP: {})".format(
                           utility.get_last_model(model_path), utility.get_last_model(dip_model_path)),
                       wait=True)


if __name__ == '__main__':
    # model_weight = 'model_weight_45'
    model_path = './overwriting/'
    # model_weight = 'Black_DnCNN_cman_weight_8.ckpt'
    dip_model_path = './combine_weight/'
    img_test = 'sign.png'

    out_copyrightImg_path = 'out_copyrightImg'
    utility.create_folder(out_copyrightImg_path)

    # eval_all_ckpts(model_path, dip_model_path, img_test)
    eval_ckpt_and_compare(model_path, dip_model_path)

    false_trigger_img, _ = GeneratorTriggerVerificationImg(40, 40).generate_trigger_and_verification_img()
    cv2.imwrite('key_imgs/trigger_image_false.png', false_trigger_img)
    eval_different_trigger_img(model_path, dip_model_path, 'trigger_image_false.png', out_copyrightImg_path)
