import numpy as np
import cv2

def psnr(img1, img2):
    img1[img1 < 0] = 0
    img1[img1 > 255] = 255

    img2[img2 < 0] = 0
    img2[img2 > 255] = 255

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    if (len(img1.shape) == 2):
        [m, n] = img1.shape
        k = 1
    elif (len(img1.shape) == 3):
        [m, n, k] = img1.shape

    B = 8
    diff = np.power(img1 - img2, 2)
    MAX = 2 ** B - 1
    MSE = np.sum(diff) / (m * n * k)
    sqrt_MSE = np.sqrt(MSE)
    PSNR = 20 * np.log10(MAX / sqrt_MSE)

    return PSNR

def ssim(img1, img2):
    k1 = 0.01
    k2 = 0.03
    L = 255
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    img1_2 = img1 * img1
    img2_2 = img2 * img2
    img1_img2 = img1 * img2

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_2 = mu1 * mu1
    mu2_2 = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_2 = cv2.GaussianBlur(img1_2, (11, 11), 1.5)
    sigma1_2 -= mu1_2

    sigma2_2 = cv2.GaussianBlur(img2_2, (11, 11), 1.5)
    sigma2_2 -= mu2_2

    sigma12 = cv2.GaussianBlur(img1_img2, (11, 11), 1.5)
    sigma12 -= mu1_mu2

    t1 = 2 * mu1_mu2 + c1
    t2 = 2 * sigma12 + c2
    t3 = t1 * t2

    t1 = mu1_2 + mu2_2 + c1
    t2 = sigma1_2 + sigma2_2 + c2
    t1 = t1 * t2

    ssim = t3 / t1
    mean_ssim = np.mean(ssim)

    return mean_ssim