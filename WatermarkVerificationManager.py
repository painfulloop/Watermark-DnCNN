import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from WatermarkedTrainedModel import WatermarkedTrainedModel
import utility
import cv2


class WMVerificationManager:
    def __init__(self, dim_imgs):
        self.dim = dim_imgs
        self.test_filename = './dataset/test/Set12/01.png'
        img = cv2.imread(self.test_filename, 0)
        self.test_img = img


    def distance_watermarking(self, S, S_prim):
        dist = np.linalg.norm(S - S_prim) * (1 / self.dim)
        return dist

    def watermark_verification(self, S, S_prim):
        S_norm = S / 255
        S_prim_norm = S_prim / 255
        dist = self.distance_watermarking(S_norm, S_prim_norm)
        watermak_succeeded = dist <= 0.00607  # 6.07 * pow(10, -3)
        return dist, watermak_succeeded

    def calculate_dist_ver_psnr(self, model_attacked_folder='./fineTuning_weight/', model_attacked_name='fineTuned_10', image_test_fn=''):
        model = WatermarkedTrainedModel()
        model.build_model(model_name=utility.get_last_model('./overwriting/'), model_path='./overwriting/')
        opt_ver_img = model.eval(test_img='key_imgs/trigger_image.png', show_input=False)
        model.build_model(model_name=model_attacked_name, model_path=model_attacked_folder)
        new_ver_img = model.eval(test_img='key_imgs/trigger_image.png', show_input=False)
        dist, watermark_succeeded = self.watermark_verification(opt_ver_img, new_ver_img)
        img = self.test_img if image_test_fn == '' else cv2.imread(image_test_fn, 0)
        psnr = utility.psnr(img, model.eval(img, show_input=False))
        return dist, watermark_succeeded, psnr


if __name__ == '__main__':
    dim = 40 * 40
    dist, watermark_succeeded, psnr = WMVerificationManager(dim).calculate_dist_ver_psnr()
    print('distance: ', dist)
    print('psnr: ', psnr)
    print('watermark_succeeded: ', watermark_succeeded)