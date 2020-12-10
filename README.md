# Watermark-DnCNN
The code is being reorganized and updated.

## Intro

# Requirements

In order to execute this code, following programs are required:
- __Python 3.7__
- __tensorflow 1.15__ (tested, but can be either higher or slithy slower)
- __Pillow__, __OpenCV__, __Matplotlib__, __Numpy__ (for windows use max numpy version 1.19.3 becayse 1.19.4 will fail sanity check)

It is advised to install tensorflow in an environment (venv or conda) for better management.

## Fast start

In order to test the repo, execute:

1. __data_process_dncnn.py__ that will create all needed datas
2. __edge.py__ that will creatae one noise images for training
3. __black_combine_train.py__ in order to train dncnn for 6 epochs 
4. __Dip_train.py__ in order to train Prior model for 2 epochs
5. __black_combine_test.pt__ in order to show results of 2 epochs retraining on watermarked image "Mr Vision"

## Scripts details

Following python files are only used as import modules for other scripts:

- __DnCNN_model.py__: contains all functions needed to create the dncnn model (also with loss and optimizer). 
If it is runned, it will compile and run once the dncnn printing all layers and testing if allright
- __DeepPrior_black_model.py__: contains all functions needed to create Deep Prior model (also with loss and optimizer)
- __edge.py__: given the image in *./input_data/noise.png*, it will resize 40x40 and execute Sobel operator on it, saving as *./input_data/spec_gt.png* and showing them to the user. Script needed before starting training
- __Psnr.py__: contains psnr and ssim metrics functions

The remaining scripts can be launched in order to acheive various results:

- __data_process_dncnn.py__: The only one script that accepts various arguments (use -h in order to get more details).
It HAVE to be runned before everything else because will take dataset images from dataset subfolders and create *img_clean_pats.npy* in data folder.
This will be used as dataset for the other scripts.
- __black_combine_train.py__: The script that trains the dncnn.
All the necessary hyperparameters or other settings are coded inside the script (since it is normally changed from there quickly).
It will take checkpoint file from __DnCNN_weight__ folder and retrain it for 5 epochs saving one checkpoint per epoch in __overwriting__ folder.
- __Dip_train.py__: The script that trains the Deep Prior network.
It will take checkpoint file form __DnCNN_weight__ folder and retrain it for 2 epochs saving one checkpoint per epoch in __combine_weight__ folder.
- __black_combine_test.py__: The script that tests the dncnn and DIP networks on 'Mr. Vision' image (watermark).
It will take dncnn and DIP checkpoints trained before and run tensorflow session over *input_data/spec_input.png* image.
Resulting image will be degraded a lot in order to see the watermarking test fail on a retrained model.
