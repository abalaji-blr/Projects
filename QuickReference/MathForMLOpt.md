# Math For ML - Optimization

[TOC]

## What is optimization?

Optimizaition

 * Is the process finding the **optimal (best) solution** from **all feasible solutions.**

 * Usually requires an optimization algorithm

 * It's applicable to many disciplines.

   

## Types of optimization

Broadly, they are classified into two categories based on the **input (aka design) variables**.

* **Discrete Optimization** : 

  The **inputs** variables are integers (aka discrete). The Discrete optimization can be solved by one of the following ways:

  * **Coimbinatorial optimization**

    * Finding the optimal object from the **finite set** of objects.
    * The feasible solution is **discrete**.
    * Examples: **Travelling Salesman Problem, Minimum Spanning Tree**, etc.
    * The **coimbinatorial optimization problem** can be defined as follows:
      * It's a problem with 4-tuple **(I, f, m, g)**
        * Where **I** is a set of instances
        * Given an instance $$x \in I, f(x) is a set of feasible solutions;$$
        * Given an instance $$x$$ and the feasible solution $$y \ of\  x $$ , $$m(x,y)$$ denotes the measure of y.
        * $$g$$ is the **goal** function, which is either *min or max* .

  * Integer Programming

  * Constraint Programming

    

* **Continuous Optimization**:

  The input varialbes are **continuous**. The standard form of continuous optimization problem is represented as:

  $$
  minimize_x	 \ \ \  f(x) \\
  subject \  to \ \  g_i(x) <= 0, 1= 1,2 ...m \\
  h_j(x) = 0, \ \ j=1, 2,...p
  $$
  Where f(x) is the **objective function**

  $$g_i(x) $$ is **inequality constraints**

  $$h_j(x)$$ is the **equality constraints**

  m >= 0 and p >= 0.

  If m and p = 0, then it is an **unconstrained optimization** problem.

  By standard form defines the **minimization** problem. For **maximization** problem, **negate** the objective function.

---



## How to Solve Continuous unconstrained optimization problem?

The input / design variables are continuous. Thus, the objective function is differentiable. Hence we have access to **gradient** at each location in space to find the optimum value.

By convention, most objective function in Machine Learning is to be **minimized**, that is the best values is the **minimum** value. Thus finding the minumum value is like finding **valleys** of the objective function and the **gradients** points uphill. The idea is to move **downhill** (opposite to the gradient).

* **Gradient Descent**

  It is a **first-order** optimization problem. i.e., it uses the first derivative - **gradient** to determine the step to find the minima.

  * StepSize (aka Learning Rate, LR)

    * Choosing StepSize ( LR) is important. This will impact time for convergence of the algorithm. If LR is too small, Gradient Descent will be slow.
    * There are **adaptive** algorithms available which will **rescale**  the LR at each iteration. 

  * Momentum

  * **Stochastic Gradient Descent** 

    * Computing the **gradient** is a time consuming process. Instead of computing the gradient using all training samples, calculate gradient using only a batch of samples and update. Though it is an approximate process, it run time performance is much better than the regular Gradient Descent.

    

---



## How to solve Continous constrained optimization problem?

* Key words:
  * Convex set / function / optimization
  * Concave optimization
  * Duality
  * Legrange multiplier
* Convex Set:
  * [Convex Set picture](https://www.easycalculation.com/maths-dictionary/convex_set.html)
* ![Convex Set:Picture](https://www.easycalculation.com/maths-dictionary/convex_set.html)

## Key Takeaways

* Convex optimization
* Differentiability of objective function.
* 