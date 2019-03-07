

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

## How to compute number of Convolution Parameters?



The generalized formula to compute the number of parameters for the Convolution layer is :

$$ channelsIn * kernelWidth * kernelHeight * channelsOut + channelsOut (for biases) $$

<u> For Example:</u>

```python
def my_model():
    model = Sequential()
    model.add(Convolution2D(34, 3, 3, activation='relu', input_shape=(28,28,1))) # output 26x26x34
    model.add(Convolution2D(32, 1, activation='relu'))  # output: 26x26x32 
    model.add(Convolution2D(32, 3, 3, activation='relu')) # 24x24x32 
    return model
```



```
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_4 (Conv2D)            (None, 26, 26, 34)        340       
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 26, 26, 32)        1120      
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 24, 24, 32)        9248      
=================================================================
```

Details:

Layer1 : 1 x 3 x 3 x 34 + 34 = 320

Layer2: 34 x 1 x 1 x 32 + 32 = 1120

Layer3: 32 X 3 x 3 x 32 + 32 = 9248

## Activation Function

The Artificial Neural Network tries to mimic the functioning of the *Neuron*. 

Basically, for the given inputs and their weights, the activation function computes the **weighted sum** of the inputs and determines the output value. 

The activation function can be linear or non-linear. In reality, most of the decisions are non-linear in nature.

The different activation functions are:

 * sigmoid

 * Hyper tangent (tanh)

 * Rectified Linear Unit (ReLu) 

 * and more.

   [Different Activation Functions](https://goo.gl/images/sUp74c) [PNG Credits: TowardsDataScience]

   

   ![Activation Function](https://github.com/abalaji-blr/Projects/blob/master/QuickReference/ActivationFunction_TowardsDataScience.png?raw=true)

   

    

## Epochs

While training the neural network, the **Epoch** defines how many times the input from the training dataset (image) is seen.

If the Epoch is 1, then the neural network will be trained by looking at all the inputs from the training dataset only once.

The **epoch** is a parameter to the model training process. Note that **higher the number of an epoch, the training time will be more**.



## Feature Maps

For a given image, when we apply the convolution filters (kernels), the *convolved output* is called  **Feature Map**. It is also known as **Activation Map**.

In general, many numbers (say K ) of filters are applied for a given image, which results in many numbers of (in this case, K) feature maps. 

The **Feature Maps** in the initial Convolutional layer learns about the **edges/corner regions** from the image. Where as, the Deeper convolutional layers learn **textures, shapes, part of objects, objects, and scene**.



## Receptive Field

The receptive field in CNN refers to the size of the input image the neuron is influenced by.

If we take the first convolution layer with filter size 3x3, the convolued output is connected to a neuron. The receptive field in this case is 3x3. It's also called as *local receptive field*.

When we stack multiple layers, the neuron in the hidden layers can see / influenced by the bigger part of the input image when compared to the initial layers. Say, if the hidden layer uses 3x3 convolution, the *local receptive field* is still 3x3. But the *effective receptive field* is the combination of the filter sizes plus the impact of the pooling layer (based on filter size, stride and padding).

<u>Example 1:</u> Let us assume,  we are applying 3x3 Convolution on the input image. What is the local and effective receptive field?

Both are 3x3.

<u>Example 2:</u> Let us assume, we are applying 2 sets of 3x3 convolution on the input image. What is the local and effective receptive field for both the layers.

* For layer1 : 
  * local and effective receptive field is 3x3.

* For layer2: 
  * Local receptive field is 3x3 (kernel size). 
  * Global receptive field is 5x5. Reason is 3x3 followed by another 3x3. Effectively, the neuron which takes the output from the layer2, will see the region of size 5x5 of the input image. 

[For more info: Refer Paper: What are Receptive Field, Effective Receptive Field ](https://arxiv.org/abs/1705.07049)



---

