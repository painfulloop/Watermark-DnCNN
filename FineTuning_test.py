# -*- coding: utf-8 -*-
import DnCNN_test
import Watermark_test as watermarkingTest
import utility
from ExecuteVerification import ExecuteVerification

if __name__ == '__main__':
    img_logo_original = watermarkingTest.eval(DnCNN_model_name='model_weight_45', model_path='./DnCNN_weight/')
    # DnCNN_test.eval(model_name='model_weight_45', model_path='./DnCNN_weight/', test_img='key_imgs/trigger_image.png',show_imput=True)
    # utility.show_image(img0, 'warmarking')


    #eval finetuning model with original data- calculate psnr and plot image. Choose epoch you need
    model_id_epoch = 10
    model_fineTuned_name = 'fineTuningDnCNNcman_weight_' + str(model_id_epoch)

    # use this to take last weight as finetuning
    # model_fineTuned_name = utility.get_last_model('./fineTuning_weight/')#uncomment this to see last epoch

    # Visualization of watermark information under model fine-tuning attacks
    img_logo_fineTun = watermarkingTest.eval(DnCNN_model_name=model_fineTuned_name, model_path='./fineTuning_weight/')

    img_logo_watermarked = watermarkingTest.eval(DnCNN_model_name=utility.get_last_model('./overwriting/'),
                                                 model_path='./overwriting/')
    # utility.show_image(img2, 'watermarking_weight_with_keys_in_net')
    # DnCNN_test.eval(model_name=utility.get_last_model('./overwriting/'), model_path='./overwriting/')
    # DnCNN_test.eval(model_name=model_fineTuned_name, model_path='./fineTuning_weight/')

    dist, watermark_succeeded = ExecuteVerification().verificationOnFineTunedImg(model_fineTuned_name)
    print('distance: ', dist)
    print('watermark_succeeded: ', watermark_succeeded)
    text = "WM {}".format("recognized" if watermark_succeeded else "lost")
    img_results = utility.create_text_image(
        utility.create_empty_image(img_logo_watermarked.shape[0], img_logo_watermarked.shape[1]), text)

    utility.show_image(
        utility.stack_images_square([img_logo_original, img_logo_watermarked, img_logo_fineTun, img_results]),
        '1 dncnn original ,2 Watermarked, 3 FineTuned, 4 WM result: {:.4f}<=0.00607'.format(dist))
