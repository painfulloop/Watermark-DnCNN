import glob, os
import numpy as np
from PIL import Image
import PIL
import random

# the pixel value range is '0-255'(uint8 ) of training data

# macro
DATA_AUG_TIMES = 1  # transform a sample to a different sample for DATA_AUG_TIMES times

def data_augmentation(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)

def generate_patches(bat_size=100, step=40, stride=80, pat_size=40, out_name="img_clean_pats", save_dir='./data', src_dir='./dataset/Train400'):
    global DATA_AUG_TIMES
    count = 0
    filepaths = glob.glob(src_dir + '/*.png')
    # filepaths = filepaths[:10] used on debug. But why?
    print ("number of training data %d" % len(filepaths))

    scales = [1, 0.9, 0.8, 0.7]

    # calculate the number of patches
    for i in range(len(filepaths)):
        img = Image.open(filepaths[i]).convert('L')  # convert RGB to gray
        for s in range(len(scales)):
            newsize = (int(img.size[0] * scales[s]), int(img.size[1] * scales[s]))
            img_s = img.resize(newsize, resample=PIL.Image.BICUBIC)  # do not change the original img
            im_h, im_w = img_s.size
            for x in range(0 + step, (im_h - pat_size), stride):
                for y in range(0 + step, (im_w - pat_size), stride):
                    count += 1

    origin_patch_num = count * DATA_AUG_TIMES
    print (origin_patch_num)

    if origin_patch_num % bat_size != 0:
        print ("++++++++++++++++")
        numPatches = (origin_patch_num // bat_size + 1) * bat_size
    else:
        numPatches = origin_patch_num
    print ("total patches = %d , batch size = %d, total batches = %d" % \
            (numPatches, bat_size, numPatches // bat_size))

    # data matrix 4-D
    inputs = np.zeros((numPatches, pat_size, pat_size, 1), dtype="uint8")

    count = 0
    # generate patches
    for i in range(len(filepaths)):
        img = Image.open(filepaths[i]).convert('L')
        for s in range(len(scales)):
            newsize = (int(img.size[0] * scales[s]), int(img.size[1] * scales[s]))
            # print newsize
            img_s = img.resize(newsize, resample=PIL.Image.BICUBIC)
            img_s = np.reshape(np.array(img_s, dtype="uint8"),
                               (img_s.size[0], img_s.size[1], 1))  # extend one dimension

            for j in range(DATA_AUG_TIMES):
                im_h, im_w, _ = img_s.shape
                for x in range(0 + step, im_h - pat_size, stride):
                    for y in range(0 + step, im_w - pat_size, stride):
                        inputs[count, :, :, :] = data_augmentation(img_s[x:x + pat_size, y:y + pat_size, :], random.randint(0, 7))
                        count += 1
    # pad the batch
    if count < numPatches:
        to_pad = numPatches - count
        inputs[-to_pad:, :, :, :] = inputs[:to_pad, :, :, :]

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    np.save(os.path.join(save_dir, out_name), inputs)
    print ("size of inputs tensor = " + str(inputs.shape))


if __name__ == '__main__':
    # Generate numpy files for training and extra for finetuning
    generate_patches(bat_size=100,
                     step=40,
                     stride=80,
                     pat_size=40,
                     out_name='img_clean_pats',
                     save_dir='./data',
                     src_dir='./dataset/Train400')

    generate_patches(bat_size=100,
                     step=40,
                     stride=80,
                     pat_size=40,
                     out_name='img_clean_KTH_TIPS',
                     save_dir='./data',
                     src_dir='./dataset/Train_KTH_TIPS')
