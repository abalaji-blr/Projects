

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

The significance of the 1x1 convolution layer is that the **output image size remains same** in terms of width and height (with the default stride or step size being 1). **However, the depth can be different**.

In general, the 1x1 convolution is applied to **reduce the spatial dimension or depth** i.e., the number of channels. As a result, the number of trainable parameters are also reduced.

This is also called as **Pointwise Convolution** or **Network in Network** (referred in *Inception Network)*.

## 3x3 convolution

In the case of 3x3 convolution, the convolution operation is performed with a kernel of size 3x3 on the given image.

For a given image (say size **n x m**), after applying 3x3 convolution, the layer output will be of reduced size ( **n-2 x m-2 **) with the default stride or step being 1. By varying the step size, you can downsample the input image, similar to that of *pooling* operation.

The importance of 3x3 convolution is that it captures the local spatial features when compared to 1x1 convolution.

## How to compute the image size after convolution?

<u>Example:1</u>

1. Let the input image be *n x n x 1*, 28x28x1

2. Apply Convolution, filter size(f): 3x3, number of filters 10. Padding(p) = 0, stride(s) = 1 (default)

   The output image size is : $$ \frac{n+ 2p -f}{s} + 1$$

   So, output image size is: ((28 + 0 -3) / 1) + 1 => 25 + 1 => 26x26

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

* Apart from the *pooling* operation, the *Stride and Padding* has more influence on the effective receptive field. Refer to the following article and computation to identify the effective receptive field.

  [Receptive Field Computation](<https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807>)

  

---

## Pooling

The pooling operation is used **to shrink the input image representation**. It also uses filter and stride to achieve down sizing of the input image. The filters in this case, usually picks up - Max value (Max Pooling) or Avgerage of the values (Avg. Pooling) captured by the filter.

Some important points to note are:

* The pooling operation happens per channel.

  * Say, input image is: 

    28x28x3 -> Max pool, filter size 2x2, stride =2 => 14x14 => reduces the input by half.

* There is **no parameter/ weights** associated to the pooling operation.

* Usually, no padding done.



$$ Input Volume: n_h * n_w *n_c ,  (height * width  * numberOfChannels)  $$

$$Output Volume:  \lfloor\frac{n_h -f}{s} + 1 \rfloor * \lfloor\frac{n_w -f }{s} + 1 \rfloor * n_c$$



## Why Convolution is preferred over Fully connected layers?

In the case of FC layers, the number of parameters are huge.

In the case of Convolution layers, **the number of parameters are *minimal*.** They are due to

* Parameter sharing: The same filter (say, vertical line detector) is used to scan over the entire image.

* Sparse Connections.

  

## More Convoultions

### Grouped Convolution

### Depthwise Separable Convolution

[Graphical Illustration](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d)

In normal convolution, the convolution operation is performed to the complete input depth.

For example: Input size: 28x28x5, Conv2d(3x3), number of kernels:10 (which is also the output channel)

For one 3x3 covolution operation, all the channels(in this case 5) are used and *one* value is computed (i.e.., convolution is nothing but a sum of products).

In the case of Depthwise Separable Convolution, two things happen:

1. 3x3 convolution happens on each input channel separately. In this case, 5.
2. Apply 1x1 convolution on the feature map obtained thru step 1, in this case 5, and add them together. Need to repeat that process for the given number of kernels ( which is the output channel as well).

Basically, we are decoupling the input and output channel from the number of parameter calculation. Thus, the **Depthwise Separable Convolution** results in *less* number of parameters.

In this case:

1. $$ 5 * 3 *3 = 45 params$$
2. $$ 5 * 10 * 1 * 1 = 50 params$$

So, *95 params* Vs. 5 x 3 x 3x 10 => 450 params (in the case of Regular Convolution).



###  Dilated Convolution



### Transpose Convolution

It is also called as *Fractionally Strided Convolution* / *Deconvolution*.



---

---



## What is Residual Block and Residual Network?

The Deep NN faces something called *Vanishing & Exploding gradient problem*.

So, inorder to train Deeper NN, something called - **Skip Connections / Short Cut** are introduced in the network. 

The activation output  from one layer is fed into few layers deep in the network. They form a **Residual block**. The Residual Network is built by stacking those **Residual Block**s.

The intution behind - why the residual block work is that these blocks learns the **Identify function** due to the *skip connections*. So, they won't hurt the performance rather it will improve from there and help train the deeper neural networks.

*One implementation thing to note is, when we do skip connections, the output dimensions of the output layer and input dimension of the deeper layer - must match*. There are multiple ways to achieve that:

* Just paddings zero to achieve certain size in the weight matrix at receiving side would match the input volume.
* Use Same size convolution layers (same filter size and num of channels) multiple times.

## What is Inception Module?

## What is Inception Network (aka gooLeNet)?

