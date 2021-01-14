import utility
import cv2
from GeneratorTriggerVerificationImg import GeneratorTriggerVerificationImg
from ExecuteVerification import ExecuteVerification
from WatermarkedTrainedModel import WatermarkedTrainedModel
from WatermarkedVisualizerModel import WatermarkedVisualizerModel
import numpy as np
import utility

def visualize_uniqueness(model_path, dip_model_path, false_trigger_imgs, out_copyrightImg_path, test_img):
    # dimostra che date in ingresso delle trigger image diverse dall'originale non produce il watermarker
    not_copyright_imgs = []
    model = WatermarkedVisualizerModel()
    model.build_model(model_path=model_path,
                      DnCNN_model_name=utility.get_last_model(model_path),
                      DIP_model_path=dip_model_path, )
    for trigger in false_trigger_imgs:
        not_copyright_img = model.eval(trigger_image=trigger, show_imput=False)
        not_copyright_imgs.append(not_copyright_img)
    ver_img = cv2.resize(cv2.imread(test_img, 0), not_copyright_imgs[0].shape, interpolation=cv2.INTER_AREA)
    concatenate_imgs = not_copyright_imgs
    concatenate_imgs.append(ver_img)
    stack_img = utility.stack_images_square(concatenate_imgs)
    cv2.imwrite(out_copyrightImg_path + '/Stack_out_with_' + str(len(false_trigger_imgs)) + '_false_trigger_imgs.png',
                stack_img)
    utility.show_image(stack_img, title='Results with ' + str(
        len(false_trigger_imgs)) + ' false trigger images - the last is Copyright image')


def uniqueness_analysis(model, trigger_imgs, verification_imgs, n_keys, dim_imgs):
    for p in range(200, n_keys + 200, 200):
        new_verification_imgs = []
        distances_w = []
        succeeded_w = []
        for i in range(len(trigger_imgs)):
            v = model.eval(test_img=trigger_imgs[i], show_input=False)
            new_verification_imgs.append(v)
            dist, succeeded = ExecuteVerification().watermark_verification(verification_imgs[i],
                                                                           new_verification_imgs[i], dim_imgs)
            distances_w.append(dist)
            succeeded_w.append(succeeded)
        min_dist = np.min(distances_w)
        all_succeeded = all(succeeded_w)

        print('min dist with ' + str(len(trigger_imgs[:p])) + ' couples of keys: ' + str(min_dist))
        print('all keys are watermark_succeeded = ', all_succeeded)


def fine_tuning_attack_analysis(dim_imgs):
    # eval finetuning model with original data- calculate psnr and plot image. Choose epoch you need
    model_id_epoch = 10
    model_fineTuned_name = 'fineTuningDnCNNcman_weight_' + str(model_id_epoch)

    dist, watermark_succeeded = ExecuteVerification(dim_imgs).verificationOnFineTunedImg(model_fineTuned_name)
    print('distance on trigger image: ', dist)
    print('watermark_succeeded: ', watermark_succeeded)

    model_visual_unwatermarked = WatermarkedVisualizerModel()
    model_visual_unwatermarked.build_model(DnCNN_model_name='model_weight_45', model_path='./DnCNN_weight/')
    img_logo_original = model_visual_unwatermarked.eval()

    # use this to take last weight as finetuning
    # model_fineTuned_name = utility.get_last_model('./fineTuning_weight/')#uncomment this to see last epoch

    # Visualization of watermark information under model fine-tuning attacks
    model_visual_finetuned = WatermarkedVisualizerModel()
    model_visual_finetuned.build_model(DnCNN_model_name=model_fineTuned_name, model_path='./fineTuning_weight/')
    img_logo_fineTun = model_visual_finetuned.eval()

    model_visual_watermarked = WatermarkedVisualizerModel()
    model_visual_watermarked.build_model(DnCNN_model_name=utility.get_last_model('./overwriting/'),
                                         model_path='./overwriting/')
    img_logo_watermarked = model_visual_watermarked.eval()

    text = "WM {}".format("recognized" if watermark_succeeded else "lost")
    img_results = utility.create_text_image(
        utility.create_empty_image(img_logo_watermarked.shape[0], img_logo_watermarked.shape[1]), text)

    utility.show_image(
        utility.stack_images_square([img_logo_original, img_logo_watermarked, img_logo_fineTun, img_results]),
        '1 dncnn original ,2 Watermarked, 3 FineTuned, 4 WM result: {:.4f}<=0.00607'.format(dist))


def generator_n_keys(h, w, n_keys):
    trigger_imgs = []
    verification_imgs = []
    for i in range(n_keys):
        t, v = GeneratorTriggerVerificationImg(h, w).generate_trigger_and_verification_img()
        trigger_imgs.append(t)
        verification_imgs.append(v)
    return trigger_imgs, verification_imgs


if __name__ == '__main__':
    model_path = './overwriting/'
    dip_model_path = './combine_weight/'
    test_img = 'test_img/sign.png'
    trigger_img = 'key_imgs/trigger_image.png'
    out_copyrightImg_path = 'out_copyrightImg'
    utility.create_folder(out_copyrightImg_path)
    n_keys = 1000
    h_w = 40
    dim_imgs = h_w * h_w

    model = WatermarkedTrainedModel()
    model.build_model(model_name=utility.get_last_model(model_path), model_path=model_path)

    print('UNIQUENESS ANALYSIS')
    trigger_imgs, verification_imgs = generator_n_keys(h_w, h_w, n_keys)
    uniqueness_analysis(model, trigger_imgs, verification_imgs, n_keys, dim_imgs)
    visualize_uniqueness(model_path, dip_model_path, trigger_imgs[:50], out_copyrightImg_path, test_img)

    print('ROBUSTENESS ANALYSIS: FINE TUNING ATTACK')
    fine_tuning_attack_analysis(dim_imgs)
