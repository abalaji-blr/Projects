[TOC]

# About Optimizers and their parameters

The **objective** of the optimizer is to **find out optimal weights** for a model for the given inputs (in our case, they are bunch of images).

The following are the steps taken by the optimiser.

 * Step1: Start with random initial  weights.
 * Step2: Based on the inputs, identify the step_size and direction to move forward.
 * Step3: Update the weights based on the step_size.
 * Continue Step2 & Step3, till loss function is minimal.



The step_size is called as **learning rate**, denoted as **$\alpha$.**

As we approach towards the local minima, when the *learning rate* is constant, we oscillate around the local minima. In order to reach, the *local minima* faster, we can **reduce** the step size. It is called as **momentum**.



There are many optimizers:

 * **GD** (Gradient Descent)
 * **SGD** (Stochastic Gradient Descent)
    * Stochastic Gradient Descent with batch size =1
    * Stochastic Gradient Descent with batch size > 1 ( **Mini Batch SGD**)

* Using **Adaptive Learning Rate**:
   * Adam  
   * RMS Prop
   * AdaDelta, Adagrad..



## Gradient Descent

It calculates the **gradient** of the **whole dataset** and updates the weights in a direction opposite to the gradients until we find a **local minima**.



## Stochastic Gradient Descent (SGD)

Instead of the whole dataset, the **weights / parameters** are updated for a **batch** of images. This makes the training process **faster** when compared with GD.

Further, the training process can be improved with help of other **hyper parameters** - momentum and Learning Rate finder.

The following are the hyper parameters:

 * learning rate - lr  ($ \alpha$)

 * momentum ($\beta$)

 * decay

 * nestrerov

   


