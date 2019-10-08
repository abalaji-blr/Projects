[TOC]

# About Optimizers and their parameters

The **objective** of the optimizer is to **find out optimal weights** for a model for the given inputs (in our case, they are bunch of images).

The following are the steps taken by the optimiser.

 * Step1: Start with random initial  weights.
 * Step2: Based on the inputs, identify the step_size and direction to move forward.
 * Step3: Update the weights based on the step_size.
 * Continue Step2 & Step3, till loss function is minimal.



The step_size is called as **learning rate**, denoted as **$\alpha$.**

As we approach towards the local minima, when the *learning rate* is constant, we oscillate around the local minima. In order to avoid the oscillations, and **speed up** to reach local minima, another hyperparameter, **momentum** helps to achieve the same. 

Following are some of optimizers:

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

   

## RMS Prop

* Used in Deep Reinforcement Learning.
* Uses element-wise, exponentially weighted moving average of the gradient squared.



## Adam

* It's also called as **Adaptiive Moment Estimation**.
* It's extension to RMSProp.
* 

## Learning Rate

The **learning rate ( $\alpha$ ) ** is an important hyper parameter, which impacts the model training. The main question involves is, how to find out the optimal learning rate or any other strategies of LR which can speed up the training process.

If the learning rate is small, the training process will be slow. If the learning rate is huge, finding local minima is tough.

 * LR Finder
    * Run one or two epochs, with a range of LR ( min_lr, max_lr) and plot against the loss.
    * Identify the LR where the loss is minimal and use that as optimal LR.
 * LR Annealing
    * Identifying the optimal / start LR is one step. But, as training progress, unchanged LR tends to not converge.
    * So, start with higher LR and as training progress, reduce the LR.
       * Step Wise Annealing
       * Expoential Decay
       * Cosine Annealing (Jeremy Howard)
    * Use **Learning Rate Scheduler**.
 * Cyclical LR (CLR)
 * OneCycle

## Blog

* [Keras - LR Finder](<https://github.com/surmenok/keras_lr_finder/blob/master/keras_lr_finder/lr_finder.py>)
* [Optimizer - Excellent source](<http://ruder.io/optimizing-gradient-descent/index.html#whichoptimizertochoose>)
* [Paper: Don't decay the Learning rate, increase batch size.](<https://arxiv.org/pdf/1711.00489v2.pdf>)
* 