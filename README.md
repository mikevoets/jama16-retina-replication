English - [中文](https://github.com/mikevoets/jama16-retina-replication/blob/master/README.zh.md)

# Code for JAMA 2016; 316(22) Replication Study

Published article link: [doi:10.1371/journal.pone.0217541](https://doi.org/10.1371/journal.pone.0217541).

Published trained neural network models: [doi:10.6084/m9.figshare.8312183](https://doi.org/10.6084/m9.figshare.8312183).

## Abstract

We have attempted to replicate the methods and reproduce the results in _Development and validation of a deep learning algorithm for detection of diabetic retinopathy in retinal fundus photographs_, published in JAMA 2016; 316(22) ([link](https://jamanetwork.com/journals/jama/fullarticle/2588763)), using publicly available data sets. We re-implemented the main method in the original study since the source code is not available.

The original study used non-public fundus images from EyePACS and three hospitals in India for training. We used a different EyePACS data set from Kaggle. The original study used the benchmark data set Messidor-2 to evaluate the algorithm’s performance. We used another distribution of the Messidor-2 data set, since the original data set is no longer available. In the original study, ophthalmologists re-graded all images for diabetic retinopathy, macular edema, and image gradability. We have one diabetic retinopathy grade per image for our data sets, and we assessed image gradability ourselves.

We were not able to reproduce the original study’s results with publicly available data. Our algorithm’s area under the receiver operating characteristic curve (AUC) of 0.951 (95% CI, 0.947-0.956) on the Kaggle EyePACS test set and 0.853 (95% CI, 0.835-0.871) on Messidor-2 did not come close to the reported AUC of 0.99 on both test sets in the original study. This may be caused by the use of a single grade per image, or different data. This study shows the challenges of reproducing deep learning method results, and the need for more replication and reproduction studies to validate deep learning methods, especially for medical image analysis.

## Help wanted: GPU time to retrain the models

The published models were trained in 2019 with code that differed from the method described in the paper in two ways (see [#22](https://github.com/mikevoets/jama16-retina-replication/pull/22)): the ImageNet weights were most likely overwritten before training, and the weight decay of 4·10⁻⁵ was applied as RMSProp's decay rate instead. The current `train.py` follows the paper. We would like to retrain the ensemble of 10 models with it on the Kaggle EyePACS data set, to check whether the method as described reproduces the reported AUCs of 0.951 (EyePACS) and 0.853 (Messidor-2).

We do not have funding or GPUs for this. Training took about 12 to 15 hours per model on one NVIDIA GTX 1080 (see [Benchmarks](#benchmarks)), so the full ensemble needs roughly 120 to 150 hours on a GPU of that class. Current GPUs are several times faster. If you can donate GPU time or cloud credits, please [open an issue](https://github.com/mikevoets/jama16-retina-replication/issues). We will publish the new models and results here.

## Requirements

Python requirements:

- Python 3.10 to 3.13
- TensorFlow >= 2.16 (with Keras 3)
- OpenCV
- Pillow
- xlrd >= 2

Install them with:

```
$ pip install -r requirements.txt
```

Training runs on a CPU, but in practice needs a GPU. On Linux, install TensorFlow with GPU support with `$ pip install "tensorflow[and-cuda]"`.

Other requirements:

- p7zip-full
- unzip

The data set scripts (`eyepacs.sh`, `messidor.sh` and `messidor2.sh`) use GNU coreutils and are meant to be run on Linux.

NB: This code was originally written for TensorFlow 1. The version used for the published study can be found at commit [15f2af3](https://github.com/mikevoets/jama16-retina-replication/tree/15f2af3).

## Preprocessing before training

1. Download the compressed [_Kaggle_ EyePACS data set](https://www.kaggle.com/competitions/diabetic-retinopathy-detection) and place all files (i.e. train and test set and labels) in the `data/eyepacs` folder. We recommend you to use the [Kaggle API](https://github.com/Kaggle/kaggle-api).

2. Run `$ ./eyepacs.sh` to decompress and preprocess the _Kaggle_ EyePACS data set, and redistribute this set into a training and test set. Run with the `--only_gradable` flag if you want to train and evaluate with gradable images only. NB: This is a large data set, so this may take hours to finish.

   The split into a training and a test set only depends on the image names and the seed (`--seed`, default 42), and is written to `data/eyepacs/bin2/split.csv`. With the default seed, it should match `vendor/eyepacs/split.csv` (or `vendor/eyepacs/split_gradable.csv` with `--only_gradable`), and the script warns if it does not. NB: This split is not the one used for the published study. The 2019 split depended on the OpenSSL version and on the order in which the file system listed the images, and it cannot be recovered. An estimated 57% of the non-referable and 96% of the referable images in this test set were in the training or validation set of the published models, so use it to evaluate newly trained models only.

For Messidor-2:

3. Run `$ ./messidor2.sh` to download, unpack, and preprocess the [Messidor-2 data set](https://www.adcis.net/en/third-party/messidor2/). The images are downloaded from the University of Iowa, which hosts the data set on behalf of Michael D. Abramoff.

For Messidor-Original:

4. Download the [Messidor-Original data set](https://www.adcis.net/en/third-party/messidor/) and place all files in the `data/messidor` folder.

5. Run `$ ./messidor.sh` to preprocess the Messidor-Original data set. Run with the `--only_gradable` flag if you want to evaluate with gradable images only.

## Training

To start training with default settings, run `$ python train.py`. The model with the highest validation AUC is saved to `./tmp/model.keras`. Optionally specify another path with the `-sm` parameter (for example `-sm=./tmp/model-1`).

After every epoch, a checkpoint of the model, the optimizer and the early stopping state is saved to `./tmp/model_checkpoint` (the model path followed by `_checkpoint`, or the path given with `-sc`). If training is interrupted, run the same command again to resume from the last completed epoch.

As described in the study, training starts from Inception-v3 weights pre-trained on ImageNet and uses RMSProp with a learning rate of 0.001 and a weight decay of 4·10⁻⁵.

Run `$ python train.py -h` to see additional optional parameters for training with your own data set, or where to save summaries or operating threshold metrics.

## Evaluation

To evaluate or test the trained neural network on the _Kaggle_ EyePACS test set, run `$ python evaluate.py -e`. To evaluate on Messidor-2, use the `-m2` flag. To evaluate on Messidor-Original, run it with the `-m` flag.

To create an ensemble of networks and evaluate the linear average of predictions, use the `-lm` parameter. To specify multiple models to evaluate as an ensemble, the model paths should be comma-separated or match a glob pattern. For example: `-lm=./tmp/model-1,./tmp/model-2,./tmp/model-3` or `-lm=./tmp/model-?`.

### Evaluate the published models

The trained models used in this study are publicly accessible and can be downloaded from [here](https://doi.org/10.6084/m9.figshare.8312183). These are TensorFlow 1 checkpoints (`model-N.index`, `model-N.meta` and `model-N.data-00000-of-00001`). `evaluate.py` loads them directly: place the files in e.g. `./tmp` and run `$ python evaluate.py -e -lm=./tmp/model-1` (or `-lm=./tmp/model-*` for the ensemble).

NB: The batch normalization layers of the published models were not trained with moving averages of the batch statistics. These models, like the TensorFlow 1 graphs they come from, therefore normalize each batch with its own statistics, also during evaluation. Their predictions thus depend on the batch size, which defaults to 32 as in the study. `evaluate.py` also standardizes the images per image for these models, like the original evaluation script did. Models trained with the current version of `train.py` do not have these quirks.

The evaluation script outputs a confusion matrix, and specificity and sensitivity by using an operating threshold. The default operating threshold is 0.5, and can be changed with the `-op` parameter.

Run `$ python evaluate.py -h` for additional parameter options.

### Evaluate on a custom data set

To evaluate the trained neural network on a different data set, follow these steps:

1. Create a Python script or start a Python session, and preprocess and resize all images to 299x299 pixels with `resize_and_center_fundus` from `lib/preprocess.py`.

2. Create a directory with two subdirectories `image_dir/0` and `image_dir/1`. Move the preprocessed images diagnosed with referable diabetic retinopathy to `image_dir/1` and the images without rDR to `image_dir/0`.

3. Create TFRecords: `$ python create_tfrecords.py --dataset_dir=image_dir` (run with `-h` for optional parameters).

4. To evaluate, run `$ python evaluate.py -o --data_dir=image_dir`. Run with `-h` for optional parameters.

## Benchmarks

NB: The benchmarks are based on TensorFlow 1 and have not been updated.

We forked the repository for TensorFlow benchmarks ([tensorflow/benchmarks](https://github.com/tensorflow/benchmarks)) to run the benchmarks with the retinal fundus images and labels used in this study ([link to fork](https://github.com/mikevoets/benchmarks)). We further provide a Docker image [maikovich/tf_cnn_benchmarks](https://hub.docker.com/r/maikovich/tf_cnn_benchmarks/) to run the benchmarks easily.

To run the benchmarks for training with this study's data on GPUs, run the following command with Docker. Substitute `path/to/train_dir` with the path to the directory that contains the TFRecord shards with the training set. The `--num_gpu`, `--batch_size` flags can be modified according to your environment's capabilities. For other flags, see the forked repository.

```
$ nvidia-docker run --mount type=bind,source=/path/to/train_dir,target=/data maikovich/tf_cnn_benchmarks:latest --data_dir=/data --model=inception3 --data_name=retina --num_gpus=2 --batch_size=64
```

As a reference, our training speed was 91.1 (± 0.4) images per second with one NVIDIA GTX 1080. For our training set of 45 717 images (80% of the 57 146 images; the other 20% are for validation), that means approximately 8.5 minutes per epoch of training. Add half a minute validation after each epoch due to initialization and loading into memory. Total training took typically 80 to 100 epochs, so it took around 12 to 15 hours to train one model. (An earlier version of this README stated a minute per epoch and 2 to 2.5 hours per model, which does not match the measured speed.)

To run the benchmarks to measure performance of evaluating test data sets, run the above command with the `--eval` flag and substitute the mount source directory with the path to the directory containing the TFRecord shards with test data.

In this repository, we also provide a example _jinja_ file (see _benchmarks.yaml.jinja.example_) to generate a Kubernetes yaml description. This can be used to run the benchmarks in a distributed fashion on multiple nodes on a Kubernetes cluster. To see more information about how to compile the jinja file into yaml, see this [README](https://github.com/tensorflow/ecosystem/tree/master/kubernetes).
