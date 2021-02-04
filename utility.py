import math
import os
import cv2
import numpy as np
import time
import json


def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


def show_image(img, title="", wait=False):
    if img.shape[0] > 1000:
        img = ResizeWithAspectRatio(img, height=1000)
    cv2.imshow(title, img)
    time.sleep(.5)
    if wait: cv2.waitKey(0)


def create_folder(path_folder):
    try:
        os.makedirs(path_folder)
    except FileExistsError:
        print('directory {} already exist'.format(path_folder))
        pass
    except OSError:
        print('creation of the directory {} failed'.format(path_folder))
        pass
    else:
        print("Succesfully created the directory {} ".format(path_folder))
    return path_folder


def get_last_model(path:str):
    _models = [file[:-len('.ckpt.meta')] for file in sorted(os.listdir(path)) if file.endswith('.ckpt.meta')]
    return _models[-1]


def get_first_model(path:str):
    _models = [file[:-len('.ckpt.meta')] for file in sorted(os.listdir(path)) if file.endswith('.ckpt.meta')]
    return _models[0]


def stack_images_row(eval_imgs: list):
    image = np.hstack(eval_imgs)
    return image

def stack_images_square(eval_imgs: list):
    l = int(math.ceil(math.sqrt(len(eval_imgs))))
    rows = []
    for row in range(l):
        r = []
        for col in range(l):
            i = row * l + col
            if i < len(eval_imgs):
                r.append(eval_imgs[i])
            else:
                r.append(np.zeros([eval_imgs[0].shape[0], eval_imgs[0].shape[1]], dtype=np.uint8))
        rows.append(np.hstack(r))
    image = np.vstack(rows)
    return image


def create_text_image(image, text:str):
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (0, int(image.shape[1]/2))
    fontScale = 1
    color = (0, 0, 255)
    thickness = 2
    image = cv2.putText(image, text, org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
    return image


def create_empty_image(w:int, h:int):
    return np.ones([w, h], dtype=np.uint8)*255

def psnr(img1, img2):
    img1 = np.clip(img1, 0, 255)

    img2 = np.clip(img2, 0, 255)

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    if (len(img1.shape) == 2):
        m, n = img1.shape
        k = 1
    elif (len(img1.shape) == 3):
        m, n, k = img1.shape

    B = 8
    diff = np.power(img1 - img2, 2)
    MAX = 2 ** B - 1
    MSE = np.sum(diff) / (m * n * k)
    sqrt_MSE = np.sqrt(MSE)
    PSNR = 20 * np.log10(MAX / sqrt_MSE)

    return PSNR


def save_json_results(datas_json:dict, file_path:str):
    with open(file_path, 'w') as out_file:
        json.dump(datas_json, out_file, indent=4)