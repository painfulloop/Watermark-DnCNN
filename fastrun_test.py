from WatermarkedVisualizerModel import WatermarkedVisualizerModel
import numpy as np
import cv2
import os
import utility


def eval_ckpt_and_compare(model):
    eval_img = model.eval()
    ver_img = cv2.resize(cv2.imread('test_img/sign.png', 0), eval_img.shape, interpolation=cv2.INTER_AREA)
    psrn_ = utility.psnr(eval_img, ver_img)
    print('psnr: ', psrn_)
    utility.show_image(np.hstack([eval_img, ver_img]),
                       title="Out: left - Original Watermarker: Right (DNCNN: {}, DIP: {})".format(
                           utility.get_last_model(model_path), utility.get_last_model(dip_model_path)),
                       wait=True)


def eval_all_ckpts(model_path, dip_model_path, img_test):
    ckpts = [c[:-len('.ckpt.index')] for c in sorted(os.listdir(dip_model_path)) if '.ckpt.index' in c]
    eval_imgs = []

    for dip_model in ckpts:
        model = WatermarkedVisualizerModel()
        model.build_model(model_path=model_path, DnCNN_model_name=utility.get_last_model(model_path))
        eval_img = model.eval()
        eval_imgs.append(eval_img)
        cv2.imwrite(out_copyrightImg_path + '/' + dip_model + '_copyright.png', eval_img)

    ver_img = cv2.resize(cv2.imread('test_img/sign.png', 0), eval_imgs[0].shape, interpolation=cv2.INTER_AREA)
    for img in eval_imgs:
        psrn_ = utility.psnr(img, ver_img)
        print('psnr: ', psrn_)
    stack_images = utility.stack_images_row(eval_imgs)
    cv2.imwrite(out_copyrightImg_path + '/Auxiliary_visualizer_per_checkpoint_row.png', stack_images)
    cv2.imshow("Auxiliary visualizer per checkpoint", stack_images)
    # cv2.imshow("Original image",
    #            cv2.resize(cv2.imread('test_img/' + img_test, 0), eval_imgs[0].shape, interpolation=cv2.INTER_AREA))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    model_path = './overwriting/'
    dip_model_path = './combine_weight/'
    test_img = 'test_img/sign.png'
    trigger_img = 'key_imgs/trigger_image.png'
    out_copyrightImg_path = 'out_copyrightImg'
    utility.create_folder(out_copyrightImg_path)

    # uncommet to view eval per checkpoint
    # eval_all_ckpts(model_path, dip_model_path, test_img)

    model = WatermarkedVisualizerModel()
    model.build_model(model_path=model_path, DnCNN_model_name=utility.get_last_model(model_path))

    eval_ckpt_and_compare(model)
