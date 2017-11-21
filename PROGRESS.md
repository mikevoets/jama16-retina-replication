# Progress-line

## Resources
- Nvidia GeForce GTX1080 GPU
- 4 Nvidia Titan X GPU

---

## 1st approach
- convolutional layer, filter size 5px, num filters 16
- convolutional layer, filter size 5px, num filters 36
- dense layer, neurons 128

### Others
-   plotting misclassified images/visualization
-   layers UI:
    - convolutional layer with tf.nn.conv2d, strides [1,1,1,1] and max-pooling, ksize [1,2,2,1], strides [1,2,2,1] and relu
    - flatten layer
    - dense layer
-   UI for reading images and plotting

---

## 2nd approach
- 2 conv layers
- flattened layer
- 2 dense layers, activation functions: relu, and nothing
- softmax layer
- cross entropy, with tf.nn.softmax_cross_entropy_with_logits
- reduce mean

### Others
- queue for reading images and labels
- removal of corrupted JPEGs
- one-hot encoding of labels
- convert EyePacs data set to Pythonic data-set library

---

## 3rd approach
- transfer learning
- inceptionV3 as base model
- dimensionality reduction by PCA
- plot transfer values
- plot example errors
- unzip 7z, zip and tar-gz

### Flowgraph
- inceptionV3
- transfer values
- fully connected layer (relu), 1024 neurons
- softmax layer
- adam optimizer, learning rate 1e-4

### Others
-   detecting fundus contours with openCV
    - cv2.cvtColor
    - cv2.findContours(gray, RETR_EXT, CHAIN_APPROX_SIMPLE)
    - increase constrast intensity
-   centering and scale normalizing to 299x299pxl images
-   add library for preprocessing/extracting data set
-   divide training and validation set
-   TrainBatcher class interface for retrieving data sets in batches
-   print validation loss

---

## 4th approach
-   fine tuning via Keras
    - 3 epochs, dense layer 1014 neurons, freeze 172 layers
    - optimizer: SGD(lr=0.0001, momentum=0.9, categorical, cross entropy)
-   model:
    - inceptionV3
    - globalAveragePooling2D
    - dense 1024, relu
    - dense 5, softmax

### Others
- divide images into subfolders grouped by classes for Keras image generator

---

# 5th approach
- transfer learning first, then fine tuning
- optimizer: rmsprop

### Others
- hyper-optimization training: optimization of hyper-parameters
- dockerize project
- run on sigma (uninett) cluster with 4 Nvidia Titan X GPUs
