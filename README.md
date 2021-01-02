# Watermark-DnCNN
The code is being reorganized and updated.

## Intro

This repo refers to Watermarking Dnns paper you can find [here](https://ieeexplore.ieee.org/document/9093125).

# Requirements

In order to execute this code, following programs are required:
- __Python 3.7__
- __tensorflow 1.15__ (tested, but can be either higher or slithy slower)
- __Pillow__, __OpenCV__, __Matplotlib__, __Numpy__ (for windows use max numpy version 1.19.3 becayse 1.19.4 will fail sanity check)

It is advised to install tensorflow in an environment (venv or conda) for better management.

## Folders

Below scripts saves training checkpoints in various folders. Those directories are used:

- __DnCNN_weight__: contains dncnn's weights after been trained for 45 epochs. This is used as 'basic model' like the customer's model that needs to be watermarked.
- __overwriting__: contains checkpoints from retraining base dncnn model (in DnCNN_weight) for watermarking.
- __combine_weight__: contains checkpoints from retraining other models. Used as Deep Prior model (Auxiliary Visualizer).


## Fast start

In order to test the repo, execute:

1. __Preprocess_dataset_for_dncnn.py__ that will create all needed datas
2. __Watermark_train__ in order to train dncnn for 6 epochs 
4. __AuxVisualizer_train__ in order to train Prior model for 2 epochs
5. __Watermark_test__ in order to show results of 2 epochs retraining on watermarked image "Mr Vision"

## Scripts details

Following python files are only used as import modules for other scripts:

- __DnCNN_model.py__: contains all functions needed to create the dncnn model (also with loss and optimizer). 
If it is runned, it will compile and run once the dncnn printing all layers and testing if allright
- __AuxVisualizerModel.py__: contains all functions needed to create Deep Prior model (also with loss and optimizer)
- __utility.py__: contains utility functions for file store and other utilities

All the scripts can be logically splitted into those categories:
- __Watermarking training__: DnCNN_model, Watermark_train, Watermark_test
- __Attack with finetune__: fineTuning_train, fineTuning_test
- __Basic env test__: DnCNN_test
- __Auxiliary visualizer__: AuxVisualizerModel, AuxVisualizer_train, 
- __Fast run scripts__: fastrun_test, fastrun_train