# Backpropagation Step by Step

**Author: Anandan Balaji, EIP 3.0 Batch 4. **        		                *[Motivation from: HMKCode.github.io]*

***



[TOC]

 ## Overview

In the network, let us consider

* **Input Layer** with two neurons

* **Hidden Layer** with two neurons

* **Output Layer** with one neuron

  ![](https://github.com/abalaji-blr/Projects/blob/master/QuickReference/Images/nn.png?raw=true)

  

## Weights

The training objective of the Neural network is to identify the weights based on the training samples.

The initial weights are randomly initialized to some small values. For our example, let the initial weights for

*{w1, w2, w3, w4, w5, w6}* be *{ 0.1, 0.2, 0.3, 0.4, 0.21, 0.23}* respectively.

![](https://github.com/abalaji-blr/Projects/blob/master/QuickReference/Images/nn_weights.png?raw=true)



## Dataset

The dataset has one sample with **two inputs and one output.**

![](https://github.com/abalaji-blr/Projects/blob/master/QuickReference/Images/nn_dataset.png?raw=true)

Our sample is as following **inputs = [2,3] and output=[1]**.

![](https://github.com/abalaji-blr/Projects/blob/master/QuickReference/Images/nn_sample.png?raw=true)

 ## Forward Pass

With the given inputs and weights, the output will be predicted. The inputs are multiplied with weights; the results are passed to the subsequent layers.

![](https://github.com/abalaji-blr/Projects/blob/master/QuickReference/Images/nn_forward.png?raw=true)

Matrix Multiplication:

$$
\begin{bmatrix}2 &3\end{bmatrix} . 
\begin{bmatrix}0.1 & 0.3 \\0.2 & 0.4 \end{bmatrix}
=
\begin{bmatrix}0.8 & 1.8 \end{bmatrix} .
\begin{bmatrix} 0.21 \\ 0.23\end{bmatrix} 
=
\begin{bmatrix} 0.582 \end{bmatrix}
$$

Matrix Multiplication details:
$$
2*0.1 + 3*0.2 = 0.2 + 0.6 = 0.8 \\
2*0.3 + 3*0.4 = 0.6 + 1.2 = 1.8 \\
\\
0.8*0.21 + 1.8 * 0.23 = 0.168 + 0.414 = 0.582
$$

## Calculating the Error

Now, it’s time to find out how our network performed by calculating the difference between the actual output and predicted one. It’s clear that our network output, or **prediction**, is not close to **actual output**. We can calculate the difference or the error as following.

![](https://github.com/abalaji-blr/Projects/blob/master/QuickReference/Images/nn_error.png?raw=true)



## Reducing the Error

Our main goal of the training is to reduce the **error** or the difference between **prediction** and **actual output**. Since **actual output** is constant, “not changing”, the only way to reduce the error is to change **prediction** value. The question now is, how to change **prediction** value?

By decomposing **prediction** into its basic elements we can find that **weights** are the variable elements affecting **prediction** value. In other words, in order to change **prediction** value, we need to change **weights** values.

![](<http://hmkcode.github.io/images/ai/bp_prediction_elements.png>)



> The question now is **how to change\update the weights value so that the error is reduced?**
> The answer is **Backpropagation!**

## Backpropagation

**Backpropagation**, short for “backward propagation of errors”, is a mechanism used to update the **weights** using [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent). It calculates the gradient of the error function with respect to the neural network’s weights. The calculation proceeds backwards through the network.

> **Gradient descent** is an iterative optimization algorithm for finding the minimum of a function; in our case we want to minimize th error function. To find a local minimum of a function using gradient descent, one takes steps proportional to the negative of the gradient of the function at the current point.

![](<http://hmkcode.github.io/images/ai/bp_update_formula.png>)

For example, to update `w6`, we take the current `w6` and subtract the partial derivative of **error** function with respect to `w6`. Optionally, we multiply the derivative of the **error** function by a selected number to make sure that the new updated **weight** is minimizing the error function; this number is called **learning rate**.

![](<http://hmkcode.github.io/images/ai/bp_w6_update.png>)

The derivation of the error function is evaluated by applying the chain rule as following

![](<http://hmkcode.github.io/images/ai/bp_error_function_partial_derivative_w6.png>)

So to update `w6` we can apply the following formula

![](<http://hmkcode.github.io/images/ai/bp_w6_update_closed_form.png>)

Similarly, we can derive the update formula for `w5` and any other weights existing between the output and the hidden layer.

![](<http://hmkcode.github.io/images/ai/bp_w5_update_closed_form.png>)

However, when moving backward to update `w1`, `w2`, `w3` and `w4` existing between input and hidden layer, the partial derivative for the error function with respect to `w1`, for example, will be as following.

![](<http://hmkcode.github.io/images/ai/bp_error_function_partial_derivative_w1.png>)

We can find the update formula for the remaining weights `w2`, `w3` and `w4` in the same way.

In summary, the update formulas for all weights will be as following:

![](<http://hmkcode.github.io/images/ai/bp_update_all_weights.png>)

We can rewrite the update formulas in matrices as following

![](<http://hmkcode.github.io/images/ai/bp_update_all_weights_matrix.png>)

## Backward Pass

Using derived formulas we can find the new **weights**.

> **Learning rate:** is a hyperparameter which means that we need to manually guess its value.

 

$$ \triangle = prediction - actual $$

$$ \triangle = 0.582 -1 = -0.418$$

$$ a  = 0.05    $$

Let us calculate the different weigh updates (rounded to two digits)


$$
\begin{bmatrix} w5 \\ w6\end{bmatrix} =
\begin{bmatrix} 0.21 \\ 0.23\end{bmatrix} 
- (0.05) *(-0.418) * \begin{bmatrix} 0.8 \\ 1.8\end{bmatrix}=
\begin{bmatrix} 0.21 \\ 0.23\end{bmatrix} -
\begin{bmatrix} -0.016 \\ -0.037\end{bmatrix} =
\begin{bmatrix} 0.226 \\ 0.267\end{bmatrix} =
\begin{bmatrix} 0.23 \\ 0.27\end{bmatrix}
$$

$$
\begin{bmatrix} w1 && w3 \\ w2 && w4\end{bmatrix} =
\begin{bmatrix} 0.1 && 0.3 \\ 0.2 && 0.4 \end{bmatrix} 
- (0.05)*(-0.418) \begin{bmatrix} 2 \\ 3 \end{bmatrix}.
   \begin{bmatrix} 0.21 && 0.23 \end{bmatrix}
$$

$$
\begin{bmatrix} 0.1 && 0.3 \\ 0.2 && 0.4 \end{bmatrix} 
- (0.05)*(-0.418)\begin{bmatrix} 0.42 && 0.46 \\ 0.63 && 0.69 \end{bmatrix}
$$


$$
\begin{bmatrix} 0.1 && 0.3 \\ 0.2 && 0.4 \end{bmatrix} -
\begin{bmatrix} -0.008 && -0.009 \\ -0.013 && -0.014 \end{bmatrix}
= \begin{bmatrix} 0.108 && 0.309 \\ 0.213 && 0.414 \end{bmatrix} 
= \begin{bmatrix} 0.11 && 0.31 \\ 0.21 && 0.41 \end{bmatrix}
$$


Now, using the new **weights** we will repeat the forward passed

## Forward Pass again

