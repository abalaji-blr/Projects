

# Key Term Definitions



[TOC]

## Convolution

In Mathematical terms, convolution is an operation where **element-wise multiplication of the two operands (tensors) followed by a summation of them.**.

The **two operands should be of the same size**. The operand could be a vector (1D tensor) or matrix (2D tensor )  in most of the time. It could be a higher dimensional tensor.

In the case of **Computer Vision**, *Convolution* is applied on an image to have many desired effects like - **blurring, sharpening the image**, **detecting the edges** and many more.



## Filters/Kernels

**Filters** (also known as **kernels / sliding-window**) is a small matrix which goes over the image (which is a big matrix of numbers).

In the case of Deep Learning, the filters (small-matrix) are initialised with random small weights and they go over the image (big matrix). When the filters perform the convolution operation with the image, they extract the features.



## 1x1 convolution

In the case of 1x1 convolution, the convolution operation is performed with a kernel of size 1x1 for the given image.

The significance of the 1x1 convolution layer is that the **output image size remains same** in terms of width and height (with the default stride or step size being 1). However, the depth can be different.

In general, the 1x1 convolution is applied to reduce the spatial dimension or depth. As a result, the number of trainable parameters are also reduced.



## 3x3 convolution

In the case of 3x3 convolution, the convolution operation is performed with a kernel of size 3x3 on the given image.

For a given image (say size **n x m**), after applying 3x3 convolution, the layer output will be of reduced size ( **n-2 x m-2 **) with the default stride or step being 1. By varying the step size, you can downsample the input image, similar to that of *pooling* operation.

The importance of 3x3 convolution is that it captures the local spatial features when compared to 1x1 convolution.



## Activation Function

The Artificial Neural Network tries to mimic the functioning of the *Neuron*. 

Basically, for the given inputs and their weights, the activation function computes the **weighted sum** of the inputs and determines the output value. 

The activation function can be linear or non-linear. In reality, most of the decisions are non-linear in nature.

The different activation functions are:

 * sigmoid

 * Hyper tangent (tanh)

 * Rectified Linear Unit (ReLu) 

 * and more.

   [Different Activation Functions](https://goo.gl/images/sUp74c)

    

## Epochs

While training the neural network, the **Epoch** defines how many times the input from the training dataset (image) is seen.

If the Epoch is 1, then the neural network will be trained by looking at all the inputs from the training dataset only once.

The **epoch** is a parameter to the model training process. Note that **higher the number of an epoch, the training time will be more**.



## Feature Maps

For a given image, when we apply the convolution filters (kernels), the *convolved output* is called  **Feature Map**. It is also known as **Activation Map**.

In general, many numbers (say K ) of filters are applied for a given image, which results in many numbers of (in this case, K) feature maps. 

The **Feature Maps** in the initial Convolutional layer learns about the **edges/corner regions** from the image. Where as, the Deeper convolutional layers learn **textures, shapes, part of objects, objects, and scene**.

