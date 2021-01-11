import Watermark_test
import numpy as np
import cv2
import os
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


def eval_different_trigger_img(model_path, dip_model_path, false_trigger_imgs, out_copyrightImg_path, test_img):
    # eval unicity
    not_copyright_imgs = []
    for i in range(len(false_trigger_imgs)):
        not_copyright_img = Watermark_test.eval(model_path=model_path,
                                                DnCNN_model_name=utility.get_last_model(model_path),
                                                DIP_model_path=dip_model_path,
                                                trigger_image=false_trigger_imgs[i])
        not_copyright_imgs.append(not_copyright_img)
    ver_img = cv2.resize(cv2.imread(test_img, 0), not_copyright_imgs[0].shape, interpolation=cv2.INTER_AREA)
    concatenate_imgs = not_copyright_imgs
    concatenate_imgs.append(ver_img)
    stack_img = utility.stack_images_square(concatenate_imgs)
    utility.show_image(stack_img, title='Unicity: results with false trigger images - the last is Copyright image')
    cv2.imwrite(out_copyrightImg_path + '/Stack_out_with_false_trigger_imgs.png', stack_img)


if __name__ == '__main__':
    model_path = './overwriting/'
    dip_model_path = './combine_weight/'
    test_img = 'test_img/sign.png'

    out_copyrightImg_path = 'out_copyrightImg'
    utility.create_folder(out_copyrightImg_path)

    # eval_all_ckpts(model_path, dip_model_path, img_test)
    eval_ckpt_and_compare(model_path, dip_model_path)
    n_keys = 80
    for i in range(n_keys):
        path = 'key_imgs/trigger_image' + str(i) + '.png'
        if os.path.isfile(path):
            print('key_imgs/trigger_image' + str(i) + '.png exist')
        else:
            print('create keys number ' + str(i))
            trigger, verification = GeneratorTriggerVerificationImg(40, 40).generate_trigger_and_verification_img()
            cv2.imwrite('key_imgs/trigger_image' + str(i) + '.png', trigger)
            cv2.imwrite('key_imgs/verification_image' + str(i) + '.png', verification)
    false_trigger = ['trigger_image' + str(i) + '.png' for i in range(n_keys)]
    eval_different_trigger_img(model_path, dip_model_path, false_trigger, out_copyrightImg_path, test_img)
