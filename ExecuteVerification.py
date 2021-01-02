import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import DnCNN_test
import utility


def distance_watermarking(S, S_prim, dim):
    dist = np.linalg.norm(S - S_prim) * (1 / dim)
    return dist


def watermark_verification(S, S_prim, dim):
    S_norm = S / 255
    S_prim_norm = S_prim / 255
    dist = distance_watermarking(S_norm, S_prim_norm, dim)
    watermak_succeeded = dist <= 0.00607#6.07 * pow(10, -3)
    return dist, watermak_succeeded


def ExecuteVerificationOnFineTunedImg(model_attacked_name='fineTuningDnCNNcman_weight_10'):
    dim_triggerImg = 40 * 40
    opt_ver_img = DnCNN_test.eval(model_name=utility.get_last_model('./overwriting/'), model_path='./overwriting/', test_img='key_imgs/trigger_image.png', show_imput=False)
    new_ver_img = DnCNN_test.eval(model_name=model_attacked_name, model_path='./fineTuning_weight/',test_img ='key_imgs/trigger_image.png', show_imput=False)
    dist, watermark_succeeded = watermark_verification(opt_ver_img, new_ver_img, dim_triggerImg)
    return dist, watermark_succeeded


if __name__ == '__main__':
    dist, watermark_succeeded = ExecuteVerificationOnFineTunedImg()
    print('distance: ', dist)
    print('watermark_succeeded: ', watermark_succeeded)