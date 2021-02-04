import os
import cv2
from GeneratorTriggerVerificationImg import GeneratorTriggerVerificationImg
from WatermarkVerificationManager import WMVerificationManager
from WatermarkedTrainedModel import WatermarkedTrainedModel
from WatermarkedVisualizerModel import WatermarkedVisualizerModel
import numpy as np
import utility
import warnings
import tensorflow as tf

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def visualize_uniqueness(model_path, dip_model_path, false_trigger_imgs, test_img, save_images=True, show_results=True):
    # dimostra che date in ingresso delle trigger image diverse dall'originale non produce il watermarker
    out_copyrightImg_path = 'results/uniqueness/'
    if save_images:
        utility.create_folder(out_copyrightImg_path)
    not_copyright_imgs = []
    model = WatermarkedVisualizerModel()
    model.build_model(model_path=model_path,
                      DnCNN_model_name=utility.get_last_model(model_path))
    for trigger in false_trigger_imgs:
        not_copyright_img = model.eval(trigger_image=trigger, show_input=False)
        not_copyright_imgs.append(not_copyright_img)
    ver_img = cv2.resize(cv2.imread(test_img, 0), not_copyright_imgs[0].shape, interpolation=cv2.INTER_AREA)
    concatenate_imgs = not_copyright_imgs
    concatenate_imgs.append(ver_img)
    stack_img = utility.stack_images_square(concatenate_imgs)
    if save_images:
        cv2.imwrite(out_copyrightImg_path + 'Stack_out_with_' + str(len(false_trigger_imgs)) + '_false_trigger_imgs.png', stack_img)
    if show_results:
        utility.show_image(stack_img, title='Results with ' + str(
            len(false_trigger_imgs)) + ' false trigger images - the last is Copyright image')


def uniqueness_analysis(model, trigger_imgs, verification_imgs, n_keys, dim_imgs, save_images=True):
    out_copyrightImg_path = 'results/uniqueness/'
    if save_images:
        utility.create_folder(out_copyrightImg_path)
    out_datas = []
    for p in range(200, n_keys + 200, 200):
        new_verification_imgs = []
        distances_w = []
        succeeded_w = []
        for i in range(len(trigger_imgs)):
            v = model.eval(test_img=trigger_imgs[i], show_input=False)
            new_verification_imgs.append(v)
            dist, succeeded = WMVerificationManager(dim_imgs).watermark_verification(verification_imgs[i],
                                                                                   new_verification_imgs[i])
            distances_w.append(round(dist, 4))
            succeeded_w.append(succeeded)
        min_dist = np.min(distances_w)
        all_succeeded = all(succeeded_w)
        out_datas.append({"p": p, "min_dist": min_dist, "succeded": all_succeeded})
        print('min dist with ' + str(len(trigger_imgs[:p])) + ' key pairs: ' + str(min_dist))
        print('all keys are watermark_succeeded = ', all_succeeded)
    if save_images:
        utility.save_json_results({"uniqueness_Results": out_datas}, out_copyrightImg_path + "datas_uniqueness.json")


def fine_tuning_attack_analysis(dim_imgs, show_distance=True, show_Separate=False, save_images=False,
                                finetuned_folder='./fineTuning_weights_Img12', dataset_name='originalDataset'):
    # eval finetuning model with original data- calculate psnr and plot image. Choose epoch you need
    result_path = 'results/fineTuning_' + dataset_name + "/"
    if save_images:
        utility.create_folder(result_path)
    distances_out = []
    psnr_all = []
    images_out = []
    files = [c for c in (os.listdir(finetuned_folder)) if '.ckpt.index' in c]
    ep = ['10', '25', '50', '75', '100']
    for epoch in ep:
        filename = "fineTuned_" + str(epoch).zfill(2) + ".ckpt.index"
        if filename in files:
            model_fineTuned_name = "fineTuned_{}".format(epoch)
            dist, watermark_succeeded, psnr = WMVerificationManager(dim_imgs).calculate_dist_ver_psnr(
                model_attacked_folder=finetuned_folder, model_attacked_name=model_fineTuned_name)
            print("{:<10} | dist={:.5f} | WM succeded={} | psnr={:.2f}".format(model_fineTuned_name, dist, watermark_succeeded, psnr))
            distances_out.append(dist)
            psnr_all.append(psnr)
            # Visualization of watermark information under model fine-tuning attacks
            model_visual_finetuned = WatermarkedVisualizerModel()
            model_visual_finetuned.build_model(DnCNN_model_name=model_fineTuned_name, model_path=finetuned_folder)
            img_logo_fineTun = model_visual_finetuned.eval()

            if show_distance:
                img_logo_fineTun = utility.create_text_image(img_logo_fineTun, "{}={:.5f}.jpg".format(epoch, dist))
            images_out.append(img_logo_fineTun)
            if show_Separate:
                utility.show_image(img_logo_fineTun, "{}={:.5f}.jpg".format(epoch, dist), wait=False)
            if save_images:
                cv2.imwrite(result_path + "fineTuned_{}.jpg".format(epoch), img_logo_fineTun)
    if save_images:
        cv2.imwrite(result_path + 'stack_out_fineTuning.png', utility.stack_images_row(images_out))
        datas_all = [{"epoch": e, "distance": d, "psnr": p} for e, d, p in zip(ep, distances_out, psnr_all)]
        utility.save_json_results({"dataset_name": dataset_name, "finetuning_results": datas_all}, result_path + "datas_finetuning.json")
    if not show_Separate:
        utility.show_image(utility.stack_images_square(images_out),
                           'Finetuning epochs 0 ' + ' '.join(ep) + ' using ' + dataset_name)


def pruning_attack_analysis(dim_imgs, pruning_weights_path="./pruning_weights/", show_distance=True,
                            show_Separate=False, save_images=False):
    result_path = 'results/pruning/'
    if save_images:
        utility.create_folder(result_path)
    images_out = []
    distances_out = []
    psnr_all = []
    pruned_ks = [float(file[8:12]) for file in sorted(os.listdir(pruning_weights_path)) if ".ckpt.meta" in file]
    for pruned_k in pruned_ks:
        k = round(float(pruned_k), 2)
        model_pruned_name = "Pruned_k{:.2f}".format(k)
        dist, watermark_succeeded, psnr = WMVerificationManager(dim_imgs).calculate_dist_ver_psnr(
            model_attacked_folder=pruning_weights_path, model_attacked_name=model_pruned_name)
        print("{} | dist={:.5f} | WM succeded={} | psnr={:.2f}".format(model_pruned_name, dist, watermark_succeeded, psnr))
        distances_out.append(dist)
        psnr_all.append(psnr)

        # Visualization of watermark information under model pruning attacks
        model_visual_pruned = WatermarkedVisualizerModel()
        model_visual_pruned.build_model(DnCNN_model_name=model_pruned_name, model_path=pruning_weights_path)
        img_logo_pruned = model_visual_pruned.eval()

        if show_distance:
            img_logo_pruned = utility.create_text_image(img_logo_pruned, "{:.2f}={:.5f}".format(k, dist))
        images_out.append(img_logo_pruned)
        if show_Separate:
            utility.show_image(img_logo_pruned, "{:.2f}={:.5f}".format(k, dist), wait=True)
        if save_images:
            cv2.imwrite(result_path + "pruned_{:.2f}.jpg".format(k), img_logo_pruned)
    if save_images:
        cv2.imwrite(result_path + 'stack_out_pruning.png', utility.stack_images_row(images_out))
        datas_all = [{"pruning_percentage": k, "distance": d, "psnr": p} for k, d, p in zip(pruned_ks, distances_out, psnr_all)]
        utility.save_json_results({"pruning_results": datas_all}, result_path + "datas_pruning.json")
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


def unwatermarked_vs_watermarked(save_images=True, show_results=True):
    result_path = 'results/WM_vs_UNWM/'
    if save_images:
        utility.create_folder(result_path)
    model_name = ['unwatermarked model', 'watermarked model']
    model_visual_unwatermarked = WatermarkedVisualizerModel()
    model_visual_unwatermarked.build_model(DnCNN_model_name='model_weight_45', model_path='./DnCNN_weight/')
    img_logo_unwatermarked = model_visual_unwatermarked.eval()
    dist, watermark_succeeded, psnr = WMVerificationManager(dim_imgs).calculate_dist_ver_psnr(
        model_attacked_folder='./DnCNN_weight/', model_attacked_name='model_weight_45')
    print("unwatermarked model | dist={:.5f} | WM succeded={} | psnr={:.2f}".format(dist, watermark_succeeded, psnr))
    model_visual_watermarked = WatermarkedVisualizerModel()
    model_visual_watermarked.build_model(DnCNN_model_name=utility.get_last_model('./overwriting/'),
                                         model_path='./overwriting/')
    dist_w, watermark_succeeded_w, psnr_w = WMVerificationManager(dim_imgs).calculate_dist_ver_psnr(
        model_attacked_folder='./overwriting/', model_attacked_name=utility.get_last_model('./overwriting/'))
    print("watermarked model | dist={:.5f} | WM succeded={} | psnr={:.2f}".format(dist_w, watermark_succeeded_w, psnr_w))
    img_logo_watermarked = model_visual_watermarked.eval()
    images_out = [img_logo_unwatermarked, img_logo_watermarked]
    test_img = cv2.resize(cv2.imread('test_img/sign.png', 0), images_out[1].shape, interpolation=cv2.INTER_AREA)
    for i in range(len(images_out)):
        psnr_ = utility.psnr(images_out[i], test_img)
        print('psnr ' + model_name[i] + ' : ' + str(round(psnr_, 2)))
    if save_images:
        cv2.imwrite(result_path + 'Un-Watermarked.png', utility.stack_images_row(images_out))
        datas_all = {"watermarked": {"distance": dist_w, "success": bool(watermark_succeeded_w), "psnr": psnr_w},
                     "unwatermarked": {"distance": dist, "success": bool(watermark_succeeded), "psnr": psnr}}
        utility.save_json_results(datas_all, result_path + "datas_w_vs_uw.json")
    if show_results:
        utility.show_image(utility.stack_images_row(images_out), 'unwatermarked, watermarked')


def fidelity_analysis(watermarked_model_path, dataset='./dataset/test/Texture12/', save_images=True):
    # print('gap < 0.05 dB is acceptable')
    result_path = 'results/fidelity/'
    if save_images:
        utility.create_folder(result_path)
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
    if save_images:
        datas_all = [{"image": i, "psnr": p} for i, p in zip(test_images, psnr_)]
        utility.save_json_results( {"psnr_avg": avg_psnr_w, "psnr_all": datas_all}, result_path + "datas_fidelity.json")


if __name__ == '__main__':
    show_uniqueness = False
    show_robustness_finetune = False
    show_robustness_finetune_kts_dataset = False
    show_robustness_pruning = False
    show_watermarked_unwatermarked = True
    show_fidelity = False

    model_path = './overwriting/'
    dip_model_path = './combine_weight/'
    test_img = 'test_img/sign.png'
    trigger_img = 'key_imgs/trigger_image.png'
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
        visualize_uniqueness(model_path, dip_model_path, trigger_imgs[:50], test_img)

    if show_robustness_finetune:
        print('ROBUSTENESS ANALYSIS: FINE TUNING ATTACK with original dataset')
        finetuned_folder = './fineTuning_weights_Img12'
        if not (os.path.isdir(finetuned_folder)):
            print('You must first train ExceuteFineTuning.py by choosing the dataset: ./data/img_clean_KTH_TIPS.npy')
        else:
            fine_tuning_attack_analysis(dim_imgs, show_distance=False, save_images=True,
                                    finetuned_folder=finetuned_folder, dataset_name='originalDataset')

    if show_robustness_finetune_kts_dataset:
        print('ROBUSTENESS ANALYSIS: FINE TUNING ATTACK with Texture KTS dataset')
        finetuned_folder = './fineTuning_weights_KTH'  # Use this for Texture KTS dataset
        if not (os.path.isdir(finetuned_folder)):
            print('You must first train ExceuteFineTuning.py by choosing the dataset: ./data/img_clean_KTH_TIPS.npy')
        else:
            fine_tuning_attack_analysis(dim_imgs, show_distance=False, save_images=True,
                                        finetuned_folder=finetuned_folder, dataset_name='KTSDataset')

    if show_robustness_pruning:
        print('ROBUSTENESS ANALYSIS: PRUNING ATTACK')
        pruning_attack_analysis(dim_imgs, show_distance=False, save_images=True)

    if show_watermarked_unwatermarked:
        print('WATERMARKED VS UNWATERMARKED')
        unwatermarked_vs_watermarked()

    if show_fidelity:
        print('FIDELITY on dataset: ', name_dataset)
        fidelity_analysis(model_path, dataset=dataset)
