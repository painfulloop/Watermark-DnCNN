# -*- coding: utf-8 -*-
import DnCNN_test
import Watermark_test as watermarkingTest


if __name__ == '__main__':
    # DnCNN_test.eval()
    img0 = watermarkingTest.eval(DnCNN_model_name='model_weight_45', model_path='./DnCNN_weight/')
    watermarkingTest.show_watermarked_image(img0, 'warmarking')

    #eval finetuning model with original data- calculate psnr and plot image
    model_id_epoch = 10
    model_fineTuned_name = 'fineTuningDnCNNcman_weight_' + str(model_id_epoch)
    DnCNN_test.eval(model_name=model_fineTuned_name, model_path='./fineTuning_weight/')

    #Visualization of watermark information under model fine-tuning attacks
    img_fineTun = watermarkingTest.eval(DnCNN_model_name=model_fineTuned_name, model_path='./fineTuning_weight/')
    watermarkingTest.show_watermarked_image(img_fineTun, title='watermark_fineTuningAttack_10epochs')

    # DnCNN_test.eval(model_name='Black_DnCNN_cman_weight_8',model_path='./overwriting/')
    img2 = watermarkingTest.eval(DnCNN_model_name='Black_DnCNN_cman_weight_8', model_path='./overwriting/')
    watermarkingTest.show_watermarked_image(img2,'watermarking_weight_with_keys_in_net')


