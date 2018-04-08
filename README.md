# Code for JAMA 2016; 316(22) Replication Study

## Introduction

We have attempted to replicate some experiments in _Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs_ that was published in JAMA 2016; 316(22) [1]. In February 2018 the paper had 236 citations in Google Scholar. To our knowledge this presented work is the first attempt to reproduce their results. We had to replicate the method since the source code is not available. Since replication studies are uncommon in the field of deep learning, we believe our results give a general insight into the reproducibility of published deep learning methods. This repository presents the source code for this replication study, and this README file gives instructions to run the replication on your own machine.

## Requirements

Python requirements:

- Python >= 3.6
- Tensorflow >= 1.4
- OpenCV >= 1.3
- Pillow
- h5py
- xlrd

Other requirements:

- p7zip-full

## Preprocessing before training

1. Run `$ git submodule update --init` to load the [create_tfrecords](https://github.com/mikevoets/create_tfrecords) repository. This tool will convert the data sets into TFRecord files.

2. Download the compressed [_Kaggle_ EyePACS data set](https://www.kaggle.com/c/diabetic-retinopathy-detection) and place all files (i.e. train and test set and labels) in the `data/eyepacs` folder. We recommend you to use the [Kaggle API](https://github.com/Kaggle/kaggle-api).

3. Run `$ ./eyepacs.sh` to decompress and preprocess the _Kaggle_ EyePACS data set, and redistribute this set into a training and test set. Run with the `--only_gradable` flag if you want to train and evaluate with gradable images only. NB: This is a large data set, so this may take hours to finish.

4. Download the [Messidor-Original data set](http://www.adcis.net/en/Download-Third-Party/Messidor.html) and place all files in the `data/messidor` folder.

5. Run `$ ./messidor.sh` to preprocess the Messidor-Original data set. Run with the `--only_gradable` flag if you want to evaluate with gradable images only.

## Training

To start training with default settings, run `$ python train.py`. To train with stochastic gradient descent, specify the `-sgd` flag. Optionally specify the path to where models checkpoints should be saved to with the `-sm` parameter.

Run `$ python train.py -h` to see additional optional parameters for training with your own data set, or where to save summaries or operating threshold metrics.

## Evaluation

To evaluate or test the trained neural network on the _Kaggle_ EyePACS test set, run `$ python evaluate.py -e`. To evaluate on Messidor-Original, run it with the `-m` flag instead.

To create an ensemble of networks and evaluate the linear average of predictions, use the `-lm` parameter. To specify multiple models to evaluate as an ensemble, the model paths should be comma-separated or satisfy a regular expression. For example: `-lm=./tmp/model-1,./tmp/model-2,./tmp/model-3` or `-lm=./tmp/model-?`.

The evaluation script outputs a confusion matrix, and specificity and sensitivity by using an operating threshold. The default operating threshold is 0.5, and can be changed with the `-op` parameter.

Run `$ python evaluate.py -h` for additional parameter options.

### Evaluate on a custom data set

To evaluate the trained neural network on a different data set, follow these steps:

1. Create a Python script or start a Python session, and preprocess and resize all images to 299x299 pixels with `.scale_normalize` from `lib/preprocess.py`.

2. Create a directory with two subdirectories `image_dir/0` and `image_dir/1`. Move the preprocessed images diagnosed with referable diabetic retinopathy to `image_dir/1` and the images without rDR to `image_dir/0`.

3. Create TFRecords: `$ python ./create_tfrecords/create_tfrecord.py --dataset_dir=image_dir` (run with `-h` for optional parameters).

4. To evaluate, run `$ python evaluate.py -o=image_dir`. Run with `-h` for optional parameters.

## References

[1] Gulshan V, Peng L, Coram M, Stumpe MC, Wu D, Narayanaswamy A, Venugopalan S, Widner K, Madams T, Cuadros J, Kim R, Raman R, Nelson PC, Mega JL, Webster DR. Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs. [JAMA. 2016;316(22):2402–2410.](https://jamanetwork.com/journals/jama/fullarticle/2588763) doi:10.1001/jama.2016.17216
