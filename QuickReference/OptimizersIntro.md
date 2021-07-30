[TOC]

# What is Optimization?

Optimization is a process of selecting the **best feasible solution** among the possible alternatives for a given criteria.

Usually, the  **objective function** is defined and the **optimization** tries to find the **set of inputs** to the objective funtion that results in a **maximum or minimum** in funtion evaluation.

---

# How to classify Optimization Algorithm?

There are many ways to classify the optimization algorithm. One way to differentiate is - **whether the objective function is differentiable or not**. 

In other words, whether we can find **derivatives** (aka slope) or not.

In the case of multi-variate objective function, the derivatives are called as **gradient** (aka partial derivatives).

Some optimization algorithm uses - **First order derivatives**, whereas some other uses **second order derivatives** as well.

In the case of multi variate, the first order derivative (aka gradient aka partial derivative) is called as **Jacobian Matrix**, where as the **second order derivative** is called as **Hessian Matrix**.

---



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

- Momentum
- Nesterov's Acceleration.

- ### Regularization / Weight Decay

  If the model remembers the training data / features, it can work well with the training dataset but when it sees the unknown test data, the test accuracy may not be same / higher than the traning accuracy. The reason being, the **model just remembered** the training data (and technically did not learn well, also called as **overfitting**) and it **did not generalize** well for any test data. In other words, the model did not have the **capacity** to predict accurately.

  To increase the **capacity**, **regularization** is used. It's also known as **weight decay**. This will make the model **generalize** well for the unseen data. 

  Note that too much of regularization may result in **underfitting**.

  What kind of regularization available?

  - L1
  - L2

  Basically, before doing the weight update, find out the **penalty** and subtract them. There is a hyper parameter associated with it called $\lambda$.

  While defining the NN architecture, we can **explicitly** add regularization using **dropout**. At times, it can be **implicitly** implemented using **image augmentation** and **early stopping**.



Further, the training process can be improved with help of other **hyper parameters** - momentum and Learning Rate finder.

The following are the hyper parameters:

 * learning rate - lr  ($ \alpha$)
 * momentum ($\beta$)
 * decay ($\lambda$)
 * nestrerov

----



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

# Blog / Resources

* [Keras - LR Finder](<https://github.com/surmenok/keras_lr_finder/blob/master/keras_lr_finder/lr_finder.py>)
* [Optimizer - Excellent source](<http://ruder.io/optimizing-gradient-descent/index.html#whichoptimizertochoose>)
* [Paper: Don't decay the Learning rate, increase batch size.](<https://arxiv.org/pdf/1711.00489v2.pdf>)
* [How to choose Optimization Algorithm : Learning Mastery](https://machinelearningmastery.com/tour-of-optimization-algorithms/)

