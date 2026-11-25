中文 - [English](https://github.com/mikevoets/jama16-retina-replication/blob/master/README.md)

# 代码给 JAMA 2016年; 316(22) 复制研究

发表文章链接：[doi:10.1371/journal.pone.0217541](https://doi.org/10.1371/journal.pone.0217541)。

发布训练的神经网络模型：[doi:10.6084/m9.figshare.8312183](https://doi.org/10.6084/m9.figshare.8312183).

## 摘要

我们试图复制开发以及验证深度学习算法的主要方法，是用来检测在 JAMA 2016 上所发表的视网膜眼底照片中的糖尿病视网膜病变（Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs; JAMA 2016年, 316(22); [链接](https://jamanetwork.com/journals/jama/fullarticle/2588763))。我们重新实现了该方法，但是源代码不可使用，因此我们使用了公开可使用的数据集。

最初的研究是使用来自 EyePACS 和印度三家医院的非公共眼底图像而进行培训。我们使用了 Kaggle 的不同 EyePACS 数据集。最初的研究使用了基准数据集 Messidor-2 来评估算法的性能。我们使用了Messidor-2数据集的另一个分布，因为原始数据集不再可用。在起初的研究中，眼科医生重新评估了糖尿病视网膜病变，黄斑水肿和图像可分级性的所有图像。我们的数据集中每个图像有一个糖尿病视网膜病变等级，我们自己评估了图像的可分级性。

我们无法复制原版的研究。我们的算法的接收器操作曲线在 Kaggle EyePACS 测试集（AUC）为0.951 (95% CI, 0.947-0.956)，在 Messidor-2 为0.853 (95% CI, 0.835-0.871)，但是在原版的研究中，未接近报告中的AUC，0.99。这可能是由于每个图像使用一个等级或不同的超参数的设置而引起的。

## 寻求帮助：用于重新训练模型的 GPU 时间

已发布的模型是在 2019 年训练的，当时的代码在两个方面与论文中描述的方法不同（见 [#22](https://github.com/mikevoets/jama16-retina-replication/pull/22)）：ImageNet 权重很可能在训练前被覆盖，而 4·10⁻⁵ 的权重衰减被当作 RMSProp 的衰减率使用。当前的 `train.py` 遵循论文的描述。我们希望用它在 Kaggle EyePACS 数据集上重新训练 10 个模型的集成，以检验按论文描述的方法能否重现报告的 AUC：0.951（EyePACS）和 0.853（Messidor-2）。

我们没有用于此项工作的经费或 GPU。在一块 NVIDIA GTX 1080 上，每个模型的训练大约需要 12 到 15 小时（见[基准](#基准)），因此整个集成在同类 GPU 上大约需要 120 到 150 小时。较新的 GPU 要快数倍。如果您能捐赠 GPU 时间或云计算额度，请[提交 issue](https://github.com/mikevoets/jama16-retina-replication/issues)。我们会在这里发布新的模型和结果。

## 要求

Python要求：

- Python 3.10 至 3.13
- TensorFlow >= 2.16（含 Keras 3）
- OpenCV
- Pillow
- xlrd >= 2

使用以下指令安装：

```
$ pip install -r requirements.txt
```

训练可以在 CPU 上执行，但实际上需要 GPU。在 Linux 上，使用 `$ pip install "tensorflow[and-cuda]"` 安装支持 GPU 的 TensorFlow。

其他需求：

- p7zip-full
- unzip

数据集脚本（`eyepacs.sh`、`messidor.sh` 和 `messidor2.sh`）使用 GNU coreutils，应在 Linux 上执行。

注意：此代码最初是为 TensorFlow 1 编写的。发表研究所使用的版本位于 commit [15f2af3](https://github.com/mikevoets/jama16-retina-replication/tree/15f2af3)。

## 训练前的预先处理

1.下载压缩的 [_Kaggle_ EyePACS数据集](https://www.kaggle.com/competitions/diabetic-retinopathy-detection) 并将所有的文件（即训练和测试集和标签）放在 `data/eyepacs` 文件夹。我们建议您使用 [Kaggle API](https://github.com/Kaggle/kaggle-api)。

2.执行 `$ ./eyepacs.sh` 来解压缩并预先处理 _Kaggle_ EyePACS 数据集，和将此集重新分配到训练和测试集中。如果您只想使用可分级图像来进行训练和评估，使用 `--only_gradable` 标志来执行。注意：这是一个大型的数据集，因此可能需要数小时才能完成。

   训练集和测试集的划分只取决于图像文件名和随机种子（`--seed`，默认为 42），并写入 `data/eyepacs/bin2/split.csv`。使用默认种子时，划分结果应与 `vendor/eyepacs/split.csv`（使用 `--only_gradable` 时为 `vendor/eyepacs/split_gradable.csv`）一致，否则脚本会发出警告。注意：此划分并非发表的研究中所使用的划分。2019 年的划分取决于 OpenSSL 的版本以及文件系统列出图像的顺序，已无法恢复。据估计，此测试集中约 57% 的无可转诊病变图像和 96% 的可转诊病变图像曾属于已发布模型的训练集或验证集，因此此测试集只应用于评估新训练的模型。

对于 Messidor-2：

3.执行 `$ ./messidor2.sh` 来下载，解压缩和预先处理 [Messidor-2 的数据集](https://www.adcis.net/en/third-party/messidor2/)。图像从爱荷华大学下载，该大学代表 Michael D. Abramoff 托管此数据集。

对于 Messidor-Original：

4.下载 [Messidor-Original 数据集](https://www.adcis.net/en/third-party/messidor/) 并将所有的文件放在  `data/messidor` 的文件夹中。

5.执行 `$ ./messidor.sh` 来预先处理 Messidor-Original 数据集。如果只要可分级图像而进行评估，使用 `--only_gradable` 标志执行。

## 培训

使用默认设置来开始训练，并执行 `$ python train.py`。验证 AUC 最高的模型会保存到 `./tmp/model.keras`。可选使用 `-sm` 参数来指定其他路径（例如 `-sm=./tmp/model-1`）。

每个训练周期结束后，模型、优化器和提前停止状态的检查点会保存到 `./tmp/model_checkpoint`（模型路径加上 `_checkpoint`，或使用 `-sc` 指定的路径）。如果训练中断，再次运行相同的命令即可从最后完成的周期继续训练。

如研究中所述，训练从在 ImageNet 上预训练的 Inception-v3 权重开始，并使用学习率为 0.001、权重衰减为 4·10⁻⁵ 的 RMSProp。

运行 `$ python train.py -h` 以查看用于使用您自己的数据集进行培训的其他可选参数，或者保存摘要或操作阈值指标的位置。

## 评价

要在 _Kaggle_ EyePACS 测试集上评估或者测试训练神经网络，执行 `$ python evaluate.py -e`。要评估 Messidor-Original，要使用 `-m` 标志来执行它。要评估 Messidor-2，使用 `-m2` 标志。

为创建网络集合和评估预测的线性平均值，要使用 `-lm` 参数。要指定要评估为集合的多个模型，模型路径应要以逗号分隔或者符合 glob 模式。例如：`-lm =./tmp/model-1，./tmp/model-2，./tmp/model-3` 或者 `-lm =./tmp/model-？`。

### 评估已发布的模型

本研究中使用的训练模型可公开访问，可以下载; [链接](https://doi.org/10.6084/m9.figshare.8312183)。这些是 TensorFlow 1 检查点（`model-N.index`、`model-N.meta` 和 `model-N.data-00000-of-00001`）。`evaluate.py` 可以直接加载它们：将文件放在例如 `./tmp` 中，然后执行 `$ python evaluate.py -e -lm=./tmp/model-1`（或使用 `-lm=./tmp/model-*` 评估集合）。

注意：已发布模型的批量归一化层在训练时没有更新批量统计的移动平均值。因此，这些模型与其来源的 TensorFlow 1 计算图一样，在评估时也使用每个批次自身的统计值进行归一化。因此它们的预测取决于批次大小，默认值与研究中相同，为 32。对于这些模型，`evaluate.py` 也像原始评估脚本一样对每张图像进行标准化。使用当前版本 `train.py` 训练的模型没有这些问题。

评估脚本通过使用操作阈值而输出混淆的矩阵，特异性和灵敏度。默认的操作阈值为0.5，也可以使用 `-op` 参数来进行更改。

执行  `$ python evaluate.py -h` 以获取其他参数的选项。

### 评估自定义数据集

要在不同的数据集上来评估训练的神经网络，请按照下列的步骤操作：

1.创建 Python 的脚本或启动 Python 的会期，使用 `lib/preprocess.py` 中的 `resize_and_center_fundus` 来预先处理所有的图像并把其调整为299x299的像素。

2.创建一个目录包含着两个子目录 `image_dir/0` 和 `image_dir/1`。把诊断为可接受糖尿病视网膜病变的预先处理图像移动到 `image_dir/1` ，把没有rDR的图像移动到 `image_dir/0`。

3.创建 TFRecords： `$ python create_tfrecords.py --dataset_dir=image_dir`（执行 `-h` 来使用可选参数)。

4.为了评估，执行 `$ python evaluate.py -o --data_dir=image_dir`。执行 `-h` 来使用可选参数。

## 基准

注意：基准测试基于 TensorFlow 1，尚未更新。

我们将存储库分为 TensorFlow 基准测试（[tensorflow/benchmarks](https://github.com/tensorflow/benchmarks)）来执行本研究中所使用的视网膜眼底图像和标签的基准（[链接到 fork](https://github.com/mikevoets/benchmarks))。我们还提供了一个 Docker 图像 [maikovich/tf_cnn_benchmarks](https://hub.docker.com/r/maikovich/tf_cnn_benchmarks/) 是为了更加轻松地执行基准测试。

要使用本研究的 GPU 数据来执行基准训练，使用Docker 来执行以下指令。将 `path/to/train_dir` 替换成为带有测试数据的 TFRecord 分片集目录的路径。可以根据环境的功能，而修改 `--num_gpu` ，`--batch_size` 标志。对于其它的标志，请参阅分叉存储库。

```
$ nvidia-docker run --mount type=bind,source=/path/to/train_dir,target=/data maikovich/tf_cnn_benchmarks:latest --data_dir=/data --model=inception3 --data_name=retina --num_gpus=2 --batch_size=64
```

执行基准来测量评估测试数据集的性能，使用 `--eval` 标志来执行以上的指令，并用带有测试数据的 TFRecord 分片集目录的路径，来替换安装源目录。

在这存储库中，我们也提供了一个例子 _jinja_ file（请参阅 _benchmarks.yaml.jinja.example_）以产生 Kubernetes yaml 的描述。这可以使用在 Kubernetes 集群上的多个节点上以分布式方式，来执行基准测试。要查看更多有关如何将 jinja 文件编译为 yaml 的信息，请参阅 [自述文件](https://github.com/tensorflow/ecosystem/tree/master/kubernetes)。
