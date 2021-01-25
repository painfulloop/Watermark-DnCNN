import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from WatermarkedTrainedModel import WatermarkedTrainedModel
import utility


class ExecuteVerification:
    def __init__(self, dim_imgs):
        self.dim = dim_imgs

    def distance_watermarking(self, S, S_prim):
        dist = np.linalg.norm(S - S_prim) * (1 / self.dim)
        return dist

    def watermark_verification(self, S, S_prim):
        S_norm = S / 255
        S_prim_norm = S_prim / 255
        dist = self.distance_watermarking(S_norm, S_prim_norm)
        watermak_succeeded = dist <= 0.00607  # 6.07 * pow(10, -3)
        return dist, watermak_succeeded

    def verificationOnAttackedImg(self, model_attacked_folder='./fineTuning_weight/',
                                  model_attacked_name='fineTuned_10'):
        model = WatermarkedTrainedModel()
        model.build_model(model_name=utility.get_last_model('./overwriting/'), model_path='./overwriting/')
        opt_ver_img = model.eval(test_img='key_imgs/trigger_image.png', show_input=False)
        model.build_model(model_name=model_attacked_name, model_path=model_attacked_folder)
        new_ver_img = model.eval(test_img='key_imgs/trigger_image.png', show_input=False)
        dist, watermark_succeeded = self.watermark_verification(opt_ver_img, new_ver_img)
        return dist, watermark_succeeded


if __name__ == '__main__':
    dim = 40 * 40
    dist, watermark_succeeded = ExecuteVerification(dim).verificationOnAttackedImg()
    print('distance: ', dist)
    print('watermark_succeeded: ', watermark_succeeded)
