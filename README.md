# Code for JAMA 2016; 316(22) Replication Study

Published article link: [arXiv:1803.04337](https://arxiv.org/abs/1803.04337v2).

## Abstract

We have attempted to replicate the main method in _Development and validation of a deep learning algorithm for detection of diabetic retinopathy in retinal fundus photographs_ published in JAMA 2016; 316(22) ([link](https://jamanetwork.com/journals/jama/fullarticle/2588763)). We re-implemented the method since the source code is not available, and we used publicly available data sets.

The original study used non-public fundus images from EyePACS and three hospitals in India for training. We used a different EyePACS data set from Kaggle. The original study used the benchmark data set Messidor-2 to evaluate the algorithm's performance. We used the similar Messidor-Original data. In the original study, ophthalmologists re-graded all images for diabetic retinopathy, macular edema, and image gradability. There was one diabetic retinopathy grade per image for our data sets, and we assessed image gradability ourselves. Hyper-parameter settings for training and validation were not described in the original study.

We were not able to replicate the original study. Our algorithm's area under the receiver operating curve (AUC) of 0.74 on the Kaggle EyePACS test set and 0.59 on Messidor-Original did not come close to the reported AUC of 0.99 in the original study. This may be caused by the use of a single grade per image, or different hyper-parameter settings. By changing the pre-processing methods, our replica algorithm's AUC increased to 0.94 and 0.82, respectively.

## Requirements

Python requirements:

- Python >= 3.6
- Tensorflow >= 1.4
- OpenCV >= 1.3
- Pillow
- h5py
- xlrd
- matplotlib >= 2.1

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

## Benchmarks

We forked the repository for TensorFlow benchmarks ([tensorflow/benchmarks](https://github.com/tensorflow/benchmarks)) to run the benchmarks with the retinal fundus images and labels used in this study ([link to fork](https://github.com/mikevoets/benchmarks)). We further provide a Docker image [maikovich/tf_cnn_benchmarks](https://hub.docker.com/r/maikovich/tf_cnn_benchmarks/) to run the benchmarks easily.

To run the benchmarks for training with this study's data on GPUs, run the following command with Docker. Substitute `path/to/train_dir` with the path to the directory that contains the TFRecord shards with the training set. The `--num_gpu`, `--batch_size` flags can be modified according to your environment's capabilities. For other flags, see the forked repository.

```
$ nvidia-docker run --mount type=bind,source=/path/to/train_dir,target=/data maikovich/tf_cnn_benchmarks:latest --data_dir=/data --model=inception3 --data_name=retina --num_gpus=2 --batch_size=64
```

To run the benchmarks to measure performance of evaluating test data sets, run the above command with the `--eval` flag and substitute the mount source directory with the path to the directory containing the TFRecord shards with test data.

In this repository, we also provide a example _jinja_ file (see _benchmarks.yaml.jinja.example_) to generate a Kubernetes yaml description. This can be used to run the benchmarks in a distributed fashion on multiple nodes on a Kubernetes cluster. To see more information about how to compile the jinja file into yaml, see this [README](https://github.com/tensorflow/ecosystem/tree/master/kubernetes).

An overview of the original benchmark results with e.g. ImageNet data can be found [here](https://www.tensorflow.org/performance/benchmarks).
