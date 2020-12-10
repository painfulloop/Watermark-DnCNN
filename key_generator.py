import numpy as np
import cv2


class GeneratorTriggerVerificationImg:
    def __init__(self, m, n):
        """
        @param m: height of the image that the generator takes as input
        @param n: width of the image that the generator takes as input
        """
        self.m = m
        self.n = n

    def generate_trigger_and_verification_img(self):
        self.trigger_img = np.random.uniform(0, 255, (self.m, self.n))
        filter_img = cv2.Sobel(self.trigger_img, cv2.CV_64F, 1, 1, ksize=3)
        self.verification_img = np.uint8(np.abs(filter_img))
        return self.trigger_img, self.verification_img
