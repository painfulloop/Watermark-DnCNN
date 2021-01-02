from GeneratorTriggerVerificationImg import GeneratorTriggerVerificationImg
import Preprocess_dataset_for_dncnn
import Watermark_train
import AuxVisualizer_train
import cv2
import utility

if __name__ == '__main__':
    key_imgs_path = utility.create_folder('key_imgs')
    trigger_img, verification_img = GeneratorTriggerVerificationImg(40, 40).generate_trigger_and_verification_img()
    cv2.imwrite(key_imgs_path + '/trigger_image.png', trigger_img)
    cv2.imwrite(key_imgs_path + '/verification_image.png', verification_img)
    Preprocess_dataset_for_dncnn.generate_patches()
    Watermark_train.train(8)  # train model to make it watermarked
    AuxVisualizer_train.train(8)  # train auxiliary visualizer to associate verification img and copyright image
