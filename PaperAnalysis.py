import os
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
                      DIP_model_path=dip_model_path)
    for trigger in false_trigger_imgs:
        not_copyright_img = model.eval(trigger_image=trigger, show_input=False)
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
            dist, succeeded = ExecuteVerification(dim_imgs).watermark_verification(verification_imgs[i],
                                                                                   new_verification_imgs[i])
            distances_w.append(round(dist, 4))
            succeeded_w.append(succeeded)
        min_dist = np.min(distances_w)
        all_succeeded = all(succeeded_w)

        print('min dist with ' + str(len(trigger_imgs[:p])) + ' key pairs: ' + str(min_dist))
        print('all keys are watermark_succeeded = ', all_succeeded)


def fine_tuning_attack_analysis(dim_imgs, show_distance=True, show_Separate=False, save_images=False, finetuned_folder='./fineTuning_weights_Img12'):
    # eval finetuning model with original data- calculate psnr and plot image. Choose epoch you need
    if save_images:
        result_path = utility.create_folder('results/fineTuning')
    model_visual_unwatermarked = WatermarkedVisualizerModel()
    model_visual_unwatermarked.build_model(DnCNN_model_name='model_weight_45', model_path='./DnCNN_weight/')

    model_visual_watermarked = WatermarkedVisualizerModel()
    model_visual_watermarked.build_model(DnCNN_model_name=utility.get_last_model('./overwriting/'),
                                         model_path='./overwriting/')
    img_logo_watermarked = model_visual_watermarked.eval()
    distances_out = [0]
    images_out = [img_logo_watermarked]
    files = [c for c in (os.listdir(finetuned_folder)) if '.ckpt.index' in c]
    ep = ['10', '25', '50', '75', '99']
    for f in sorted(files):
        epoch = f[10:12]
        if epoch in ep:
            print(epoch)
            model_fineTuned_name = "fineTuned_{}".format(epoch)
            dist, watermark_succeeded = ExecuteVerification(dim_imgs).verificationOnAttackedImg(
                model_attacked_folder=finetuned_folder, model_attacked_name=model_fineTuned_name)
            print("{} | dist={:.5f} | WM succeded={} |".format(model_fineTuned_name, dist, watermark_succeeded))
            distances_out.append(dist)
            # Visualization of watermark information under model fine-tuning attacks
            model_visual_finetuned = WatermarkedVisualizerModel()
            model_visual_finetuned.build_model(DnCNN_model_name=model_fineTuned_name, model_path=finetuned_folder)
            img_logo_fineTun = model_visual_finetuned.eval()

            if show_distance:
                img_logo_fineTun = utility.create_text_image(img_logo_fineTun, "{}={:.5f}.jpg".format(epoch, dist))
            images_out.append(img_logo_fineTun)
            if show_Separate:
                utility.show_image(img_logo_fineTun, "{}={:.5f}.jpg".format(epoch, dist), wait=True)
            if save_images:
                cv2.imwrite(result_path + "/fineTuned_{}_{:.5f}.jpg".format(epoch, dist), img_logo_fineTun)
                cv2.imwrite(result_path + '/stack_out_fineTuning.png', utility.stack_images_row(images_out))
    test_img = cv2.resize(cv2.imread('test_img/sign.png', 0), images_out[1].shape, interpolation=cv2.INTER_AREA)
    for i in range(1, len(images_out) - 1):
        if show_distance == False:
            psnr_ = utility.psnr(images_out[i], test_img)
            print('psnr eopch ' + ep[i - 1] + ': ' + str(round(psnr_, 2)))
    if not show_Separate:
        utility.show_image(utility.stack_images_square(images_out), '1 Watermarked, other fineTuning 5, 10, 15 ...')
        # utility.show_image(utility.stack_images_row(images_out),
        #                    '1 Watermarked, other fineTuning 5, 10, 15 ...')


def pruning_attack_analysis(dim_imgs, pruning_weights_path="./pruning_weights/", show_distance=True,
                            show_Separate=False, save_images=False):
    if save_images:
        utility.create_folder('results/pruning')
    model_visual_watermarked = WatermarkedVisualizerModel()
    model_visual_watermarked.build_model(DnCNN_model_name=utility.get_last_model('./overwriting/'),
                                         model_path='./overwriting/')
    img_logo_watermarked = model_visual_watermarked.eval()

    # eval pruned model with original data- calculate psnr and plot image. Choose pruned k you need
    images_out = [img_logo_watermarked]
    distances_out = [0]
    pruned_ks = [float(file[8:12]) for file in sorted(os.listdir(pruning_weights_path)) if ".ckpt.meta" in file]
    for pruned_k in pruned_ks:
        k = round(float(pruned_k), 2)
        model_pruned_name = "Pruned_k{:.2f}".format(k)
        dist, watermark_succeeded = ExecuteVerification(dim_imgs).verificationOnAttackedImg(
            model_attacked_folder=pruning_weights_path, model_attacked_name=model_pruned_name)

        # Visualization of watermark information under model pruning attacks
        model_visual_pruned = WatermarkedVisualizerModel()
        model_visual_pruned.build_model(DnCNN_model_name=model_pruned_name, model_path=pruning_weights_path)
        img_logo_pruned = model_visual_pruned.eval()

        print("{} | dist={:.5f} | WM succeded={} |".format(model_pruned_name, dist, watermark_succeeded))
        if show_distance:
            img_logo_pruned = utility.create_text_image(img_logo_pruned, "{:.2f}={:.5f}".format(k, dist))
        images_out.append(img_logo_pruned)
        distances_out.append(dist)
        if show_Separate:
            utility.show_image(img_logo_pruned, "{:.2f}={:.5f}".format(k, dist), wait=True)
        if save_images:
            cv2.imwrite("results/pruning/pruned_{:.2f}_{:.5f}.jpg".format(k, dist), img_logo_pruned)
    test_img = cv2.resize(cv2.imread('test_img/sign.png', 0), images_out[1].shape, interpolation=cv2.INTER_AREA)
    for i in range(1, len(images_out)):
        if show_distance == False:
            psnr_ = utility.psnr(images_out[i], test_img)
            print('psnr ' + str(pruned_ks[i - 1]) + ': ' + str(round(psnr_, 2)))
    if not show_Separate:
        # utility.show_image(utility.stack_images_square(images_out), '1 Watermarked, other pruning 0.1, 0.2,...')
        utility.show_image(utility.stack_images_row(images_out), '1 Watermarked, other pruning 0.1, 0.2,...')


def generator_n_keys(h, w, n_keys):
    trigger_imgs = []
    verification_imgs = []
    for i in range(n_keys):
        t, v = GeneratorTriggerVerificationImg(h, w).generate_trigger_and_verification_img()
        trigger_imgs.append(t)
        verification_imgs.append(v)
    return trigger_imgs, verification_imgs


def unwatermarked_vs_watermarked():
    model_name = ['unwatermarked model', 'watermarked model']
    model_visual_unwatermarked = WatermarkedVisualizerModel()
    model_visual_unwatermarked.build_model(DnCNN_model_name='model_weight_45', model_path='./DnCNN_weight/')
    img_logo_unwatermarked = model_visual_unwatermarked.eval()
    dist, watermark_succeeded = ExecuteVerification(dim_imgs).verificationOnAttackedImg(
        model_attacked_folder='./DnCNN_weight/', model_attacked_name='model_weight_45')
    print("unwatermarked model | dist={:.5f} | WM succeded={} |".format(dist, watermark_succeeded))
    model_visual_watermarked = WatermarkedVisualizerModel()
    model_visual_watermarked.build_model(DnCNN_model_name=utility.get_last_model('./overwriting/'),
                                         model_path='./overwriting/')
    dist, watermark_succeeded = ExecuteVerification(dim_imgs).verificationOnAttackedImg(
        model_attacked_folder='./overwriting/', model_attacked_name=utility.get_last_model('./overwriting/'))
    print("watermarked model | dist={:.5f} | WM succeded={} |".format(dist, watermark_succeeded))
    img_logo_watermarked = model_visual_watermarked.eval()
    images_out = [img_logo_unwatermarked, img_logo_watermarked]
    test_img = cv2.resize(cv2.imread('test_img/sign.png', 0), images_out[1].shape, interpolation=cv2.INTER_AREA)
    for i in range(len(images_out)):
        psnr_ = utility.psnr(images_out[i], test_img)
        print('psnr ' + model_name[i] + ' : ' + str(round(psnr_, 2)))
    cv2.imwrite('out_copyrightImg/Un-Watermarked.png', utility.stack_images_row(images_out))
    utility.show_image(utility.stack_images_row(images_out), 'unwatermarked, watermarked')


def fidelity_analysis(watermarked_model_path, dataset='./dataset/test/Texture12/'):
    # print('gap < 0.05 dB is acceptable')
    seed = 42
    watermarked_model = WatermarkedTrainedModel()
    watermarked_model.build_model(model_name=utility.get_last_model(watermarked_model_path),
                                  model_path=watermarked_model_path, seed=seed)
    unWatermarked_model = WatermarkedTrainedModel()
    unWatermarked_model.build_model(seed=seed)
    unwatermarked_imgs_out = []
    psnr_ = []
    test_images = [dataset + img for img in sorted(os.listdir(dataset))]
    for img in test_images:
        unwatermarked_imgs_out.append(unWatermarked_model.eval(test_img=img, show_input=False))
    for index, out in enumerate(unwatermarked_imgs_out):
        test_img = cv2.imread(test_images[index], 0)
        psnr_.append(utility.psnr(out, test_img))
    avg_psnr = sum(psnr_) / len(psnr_)
    print('UnWatermaked model | avg psnr: ', avg_psnr)
    watermarked_imgs_out = []
    psnr_w = []
    for img in test_images:
        watermarked_imgs_out.append(watermarked_model.eval(test_img=img, show_input=False))
    for index, out in enumerate(watermarked_imgs_out):
        test_img = cv2.imread(test_images[index], 0)
        psnr_w.append(utility.psnr(out, test_img))
    avg_psnr_w = sum(psnr_w) / len(psnr_w)
    print('Watermarked model | avg psnr: ', avg_psnr_w)


if __name__ == '__main__':
    show_uniqueness = False
    show_robustness_finetune = True
    show_robustness_pruning = False
    show_watermarked_unwatermarked = False
    show_fidelity = False

    model_path = './overwriting/'
    dip_model_path = './combine_weight/'
    test_img = 'test_img/sign.png'
    trigger_img = 'key_imgs/trigger_image.png'
    out_copyrightImg_path = 'out_copyrightImg'
    utility.create_folder(out_copyrightImg_path)
    dataset_partial_path = './dataset/test'
    name_dataset = 'BSD68/'
    dataset = os.path.join(dataset_partial_path, name_dataset)

    n_keys = 1000
    h_w = 40
    dim_imgs = h_w * h_w

    model = WatermarkedTrainedModel()
    model.build_model(model_name=utility.get_last_model(model_path), model_path=model_path)

    if show_uniqueness:
        print('UNIQUENESS ANALYSIS')
        trigger_imgs, verification_imgs = generator_n_keys(h_w, h_w, n_keys)
        uniqueness_analysis(model, trigger_imgs, verification_imgs, n_keys, dim_imgs)
        visualize_uniqueness(model_path, dip_model_path, trigger_imgs[:50], out_copyrightImg_path, test_img)

    if show_robustness_finetune:
        print('ROBUSTENESS ANALYSIS: FINE TUNING ATTACK')
        finetuned_folder = './fineTuning_weights_Img12'
        #finetuned_folder = './fineTuning_weights_KTH'# Use this for Texture KTS dataset
        fine_tuning_attack_analysis(dim_imgs, show_distance=False, show_Separate=False, save_images=True, finetuned_folder=finetuned_folder)

    if show_robustness_pruning:
        print('ROBUSTENESS ANALYSIS: PRUNING ATTACK')
        pruning_attack_analysis(dim_imgs, show_distance=False, show_Separate=False, save_images=True)

    if show_watermarked_unwatermarked:
        print('WATERMARKED VS UNWATERMARKED')
        unwatermarked_vs_watermarked()

    if show_fidelity:
        print('FIDELITY on dataset: ', name_dataset)
        fidelity_analysis(model_path, dataset=dataset)
