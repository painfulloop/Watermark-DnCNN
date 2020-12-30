import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import DnCNN_test


def distance_watermarking(S, S_prim, dim):
    dist = np.linalg.norm(S - S_prim) * (1 / dim)
    return dist


def watermark_verification(S, S_prim, dim):
    S_norm = S / 255
    S_prim_norm = S_prim / 255
    dist = distance_watermarking(S_norm, S_prim_norm, dim)
    if dist <= 6.07 * pow(10, -3):
        watermak_succeeded = True
    else:
        watermak_succeeded = False
    return dist, watermak_succeeded


if __name__ == '__main__':
    dim_triggerImg = 40 * 40
    opt_ver_img = DnCNN_test.eval(test_img='key_imgs/trigger_image.png' )
    new_ver_img = DnCNN_test.eval(model_name='fineTuningDnCNNcman_weight_10', model_path='./fineTuning_weight/',test_img ='key_imgs/trigger_image.png')
    dist, watermark_succeeded = watermark_verification(opt_ver_img, new_ver_img, dim_triggerImg)
    print('distance: ', dist)
    print('watermark_succeeded: ', watermark_succeeded)
