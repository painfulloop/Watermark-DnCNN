import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from WatermarkedTrainedModel import WatermarkedTrainedModel
import utility

class ExecuteVerification:
    def __init__(self):
        pass

    def distance_watermarking(self, S, S_prim, dim):
        dist = np.linalg.norm(S - S_prim) * (1 / dim)
        return dist

    def watermark_verification(self, S, S_prim, dim):
        S_norm = S / 255
        S_prim_norm = S_prim / 255
        dist = self.distance_watermarking(S_norm, S_prim_norm, dim)
        watermak_succeeded = dist <= 0.00607  # 6.07 * pow(10, -3)
        return dist, watermak_succeeded

    def verificationOnFineTunedImg(self, model_attacked_name='fineTuningDnCNNcman_weight_10'):
        dim_triggerImg = 40 * 40
        model = WatermarkedTrainedModel()
        model.build_model(model_name=utility.get_last_model('./overwriting/'), model_path='./overwriting/')
        opt_ver_img = model.eval(test_img='key_imgs/trigger_image.png', show_imput=False)
        model.build_model(model_name=model_attacked_name, model_path='./fineTuning_weight/')
        new_ver_img = model.eval(test_img='key_imgs/trigger_image.png', show_imput=False)
        dist, watermark_succeeded = self.watermark_verification(opt_ver_img, new_ver_img, dim_triggerImg)
        return dist, watermark_succeeded

    def verificationMoreTriggerImgs(self, trigger, triggers_false, model_path='./overwriting/'):
        dim_triggerImg = 40 * 40
        print('dim', trigger.shape)
        model = WatermarkedTrainedModel()
        model.build_model(model_name=utility.get_last_model(model_path), model_path=model_path)
        opt_ver_img = model.eval(test_img=trigger, show_imput=False)
        result = []
        for img in triggers_false:
            new_ver_img = model.eval(test_img=img, show_imput=False)
            dist, watermark_succeeded = self.watermark_verification(opt_ver_img, new_ver_img, dim_triggerImg)
            result.append(dist, watermark_succeeded)
        return result


if __name__ == '__main__':
    dist, watermark_succeeded = ExecuteVerification().verificationOnFineTunedImg()
    print('distance: ', dist)
    print('watermark_succeeded: ', watermark_succeeded)
