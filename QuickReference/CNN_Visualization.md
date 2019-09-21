[TOC]

# Kernel, Channel Visualization

## How to find out what our models (keras) are doing?

* **Sailency** (aka importance, significance)

  * [What is Sailency?](<https://raghakot.github.io/keras-vis/visualizations/saliency/>)
  * **From the classified output(ie., class), find out the gradient of the output category with respect to the input image. This will tell us, what is the change in the output value to a small change in input image.**
  * [Paper: Deep Inside CNN : Visualizing Models and Sailency Maps](<https://arxiv.org/pdf/1312.6034v2.pdf>)
  * **Drawback with Sailency** : we have to make small changes (pixels) to the input image to understand the impact on the output (i.e.., class determination). We have to try out many number to times. 

*  **Guided / Rectified Sailency**

  * In Guided Sailency, during back-propagation, modify to allow only positive gradients.

* **Activation Maximization**

  * Keep the weights and outputs constant but modify the input such that it maximises certain neutrons.

* **Global Average Pooling (GAP)**

  * Instead of using Fully Connected Layer (FC/ Dense), we can use Global Average Pool layer.
  * The drawback of FC/Dense layer is that, varying size input images can't be passed in.
    * Also, increases the param count.
  * In the case of GAP:
    * Takes **average of each feature map** and the resulting vector can be fed to **softmax** layer.
    * No **parameters** => No **overfitting**.

* **Class Activation Maximization /Mapping**

* **GradCAM ( Gradient weighted Class Action Map)**

  * Applicable to any CNN network. **Don't need to change the network with GAP layers**.

  * **Don't need re-training**.

  * The idea is to look at the **final convolution layer output**, which produces the localisation map, which **highlights the important regions** for predicting the class.

  * Steps:

    * Load the pre-trained model
    * Load the input image
    * Collect the predicted class (top most class index).
    * **Take the output of the last convolutional layer**.
    * **Compute the gradient of the class output value w.r.t to N feature maps**.
    * **Pool the gradients over all axes leaving out the channel dimension**.
      * This is equivlaent to Global Average pooling of all the gradients for a channel.
      * That is, you get *one avg gradient value* for each channel. 
      * If there is N number of channels in that layer, then you get a **vector of N values**.
    * The **pooling of gradients** is the weights. Multiply the vector of pooled gradients with the feature map of last convolutional layer.
    * Get only the postitive values. (Basically, send those values to ReLU).
    * Average the weighted feature maps along the channels.
    * Normalize the heat map to make values between 0 and 1.

    

* **DeconvNet**

* **Unpooling Layers**

## Tools / Framework - Interpretability

* [Github: Collection of tools for Network Interpretability ](<https://github.com/tensorflow/lucid#activation-atlas-notebooks>)
* [Tensor Space | 3D Visualization Framework](<https://tensorspace.org/index.html>)
* 

