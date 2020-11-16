# Math for ML - Vector Calculus

[TOC]

## Introduction

**Calculus** is a study of how things changes.

**Vector Calculus or vector analysis** is concerned with the **differentiation (sensitivity to change) or integration** (combining the infinitesimal of data to describe displacement, area, volume etc.) of vector fields. 

A **function** f is a quantity which relates two quantities to each other.

A **differential function** of a real variable is a function whose **derivative (which measures the steepness or slope) ** exists for each point in the domain.  In other words, **derivative** is a **rate of change of a function** with respect to a variable.

## Differentiation



**Differentiation** is a process of calculating the **rate of change of one variable with respect to another variable.**

The **derivatives** are written as $$ \frac{dy}{dx} or f^{'}$$.

A **function** is **differential** if the function is both **continuous** and has only one output for every input. Note that the converse is not true. That is, every continuous function is not necessarily differential function.

The **Taylor Series** is a representation of a function f as an **infinite** sum of terms. The terms are determined using **derivatives** of f. It is a **series ** used to create an estimate (guess) what a function looks like. There is a special kind of **Taylor Series** called **Maclaurin Series**. If any function is infinitely differentiable  at x, it has a **Taylor series** at x. 

## Gradient



A function may depend on **one or more** variables. The **generialization** of  the derivative to functions of serveral variables is called **gradient**. That is, we find the **gradient** of the function f ($${ f(x) = f(x1, x2 …) }$$ ) with respect to x by varing one variable at a time and keeping the others constant. Thus **gradient** is the collection of **partial derivatives**.

## Jacobian 

The **collection of first-order partial derivatives** of a function is called **Jacobian**.

Geometrically, the **determinant of Jacobian matrix** gives the **magnification / scaling factor** when we transform the area or volume.

If we compute the **gradient** of $$m X n$$ matrix of A with respect to $$ p X q $$ matrix of B, then the resulting Jacobian would be $$(m Xn)\  X \  (pXq)$$ i.e.., the four dimensional tensor of J.

## Back Propagation and Automatic Differentiation

**Symbolic Computation** means computing the expression represented using **symbols**. That is, expression using non-numerical values ie., **symbols** like algebra.

**Numeric Computation** means computing the expression represented using **numbers**.

The **Automatic Differentiation** is a set of techniques to numerically compute the gradient of a function by working with **intermediate values and chain rule**.

The **Back Propagation** in Deep Neural Networks can make use of **automatic differentiation** for gradient computation.



## Hessian 

**Hessian** is the collection of **second-order** partial derivatives.





## To Do

* Lagrange Multipliers
* Laplace approximations.

## References

* [Taylor Series - Math is Fun](https://www.mathsisfun.com/algebra/taylor-series.html)
* 