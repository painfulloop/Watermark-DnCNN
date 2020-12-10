from key_generator import GeneratorTriggerVerificationImg
import data_process_dncnn
import black_combine_train
import Dip_train
import black_combine_test
import cv2
import utility

key_imgs_path = utility.create_folder('key_imgs')
trigger_img, verification_img = GeneratorTriggerVerificationImg(40, 40).generate_trigger_and_verification_img()
cv2.imwrite(key_imgs_path + '/trigger_image.png', trigger_img)
cv2.imwrite(key_imgs_path + '/verification_image.png', verification_img)
data_process_dncnn.generate_patches()
black_combine_train.train() #train model to make it watermarked
Dip_train.train() #train auxiliary visualizer to associate verification img and copyright image






