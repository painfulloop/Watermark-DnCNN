import Watermark_test
import numpy as np
import cv2
import os
import utility
from GeneratorTriggerVerificationImg import GeneratorTriggerVerificationImg
from ExecuteVerification import ExecuteVerification
import DnCNN_test


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


def visualize_uniqueness(model_path, dip_model_path, false_trigger_imgs, out_copyrightImg_path, test_img):
    # dimostra che date in ingresso delle trigger image diverse dall'originale non produce il watermarker
    not_copyright_imgs = []
    for trigger in false_trigger_imgs:
        not_copyright_img = Watermark_test.eval(model_path=model_path,
                                                DnCNN_model_name=utility.get_last_model(model_path),
                                                DIP_model_path=dip_model_path,
                                                trigger_image=trigger)
        not_copyright_imgs.append(not_copyright_img)
    ver_img = cv2.resize(cv2.imread(test_img, 0), not_copyright_imgs[0].shape, interpolation=cv2.INTER_AREA)
    concatenate_imgs = not_copyright_imgs
    concatenate_imgs.append(ver_img)
    stack_img = utility.stack_images_square(concatenate_imgs)
    cv2.imwrite(out_copyrightImg_path + '/Stack_out_with_' + str(len(false_trigger_imgs)) + '_false_trigger_imgs.png',
                stack_img)
    utility.show_image(stack_img, title='Results with ' + str(
        len(false_trigger_imgs)) + ' false trigger images - the last is Copyright image')


def uniqueness_analysis(trigger_imgs, verification_imgs, dim_imgs, model_path):
    new_verification_imgs = []
    distances_w = []
    succeeded_w = []

    for i in range(len(trigger_imgs)):
        v = DnCNN_test.eval(model_name=utility.get_last_model(model_path), model_path=model_path,
                            test_img=trigger_imgs[i], show_imput=False)
        new_verification_imgs.append(v)
        dist, succeeded = ExecuteVerification().watermark_verification(verification_imgs[i], new_verification_imgs[i],
                                                                       dim_imgs)
        distances_w.append(dist)
        succeeded_w.append(succeeded)
    min_dist = np.min(distances_w)
    all_succeeded = all(succeeded_w)
    return min_dist, all_succeeded


if __name__ == '__main__':
    model_path = './overwriting/'
    dip_model_path = './combine_weight/'
    test_img = 'test_img/sign.png'
    trigger_img = 'key_imgs/trigger_image.png'
    out_copyrightImg_path = 'out_copyrightImg'
    utility.create_folder(out_copyrightImg_path)

    # eval_all_ckpts(model_path, dip_model_path, img_test)
    eval_ckpt_and_compare(model_path, dip_model_path)

    # TODO:se vogliamo salvare le varie keys generate
    # for i in range(n_keys):
    #     path = 'key_imgs/trigger_image' + str(i) + '.png'
    #     if os.path.isfile(path):
    #         print('key_imgs/trigger_image' + str(i) + '.png exist')
    #     else:
    #         print('create keys number ' + str(i))
    #         trigger, verification = GeneratorTriggerVerificationImg(40, 40).generate_trigger_and_verification_img()
    #         cv2.imwrite('key_imgs/trigger_image' + str(i) + '.png', trigger)
    #         cv2.imwrite('key_imgs/verification_image' + str(i) + '.png', verification)
    # false_trigger = ['trigger_image' + str(i) + '.png' for i in range(n_keys)]
    # false_verification = ['verification_image' + str(i) + '.png' for i in range(n_keys)]

    n_keys = 800
    trigger_imgs = []
    verification_imgs = []
    h_w = 40
    for i in range(n_keys):
        t, v = GeneratorTriggerVerificationImg(h_w, h_w).generate_trigger_and_verification_img()
        trigger_imgs.append(t)
        verification_imgs.append(v)

    dim_img = h_w * h_w
    for i in range(200, n_keys + 200, 200):
        print(i)
        minimal_dist, all_succeeded = uniqueness_analysis(trigger_imgs[:i], verification_imgs[:i], dim_img, model_path)
        print('min dist with ' + str(len(trigger_imgs[:i])) + ' couples of keys: ' + str(minimal_dist))
        print('all keys are watermark_succeeded = ', all_succeeded)

    visualize_uniqueness(model_path, dip_model_path, trigger_imgs[:50], out_copyrightImg_path, test_img)
