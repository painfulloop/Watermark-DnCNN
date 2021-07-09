from GeneratorTriggerVerificationImg import GeneratorTriggerVerificationImg
import Preprocess_dataset_for_dncnn
import Watermark_train
import AuxVisualizer_train
import cv2
import utility
import os

if __name__ == '__main__':
    key_imgs_path = utility.create_folder('key_imgs')
    trigger_path = os.path.join(key_imgs_path, 'trigger_image.png')
    verification_path = os.path.join(key_imgs_path, 'verification_image.png')
    if not (os.path.isfile(trigger_path) or os.path.isfile(verification_path)):
        trigger_img, verification_img = GeneratorTriggerVerificationImg(40, 40).generate_trigger_and_verification_img()
        cv2.imwrite(trigger_path, trigger_img)
        cv2.imwrite(verification_path, verification_img)
    Preprocess_dataset_for_dncnn.generate_patches()
    Watermark_train.train(epochs=8, trigger_img=key_imgs_path + '/trigger_image.png',
                          verification_img=key_imgs_path + '/verification_image.png')  # train model to make it watermarked
    AuxVisualizer_train.train(epochs=8)  # train auxiliary visualizer to associate verification img and copyright image
