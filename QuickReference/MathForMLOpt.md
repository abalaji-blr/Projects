# Math For ML - Optimization

[TOC]

## What is optimization?

Optimizaition

 * Is the process finding the **optimal (best / most favourable solution to objective function) solution** from **all feasible solutions.**

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
        * Given an instance $$x \in I, f(x)$$ is a set of feasible solutions;
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
  * Duality
  * Legrange multiplier

### Convex and Concave Set

Note: Don't confuse with the word **convex / concave** with respect to physics ie., optics, convex and concave lenses.

In Math, specially in geometry, when you join the two points in the set and the line segment also can reside completely in the set. Thus, when all line segments reside completely between any pair of points in the set, then it is called **convex set**. 

If the set **does not contain all line segments** (ie., some part of the line segment is not in the set), then it is called as **concave set**.

<blockquote class="embedly-card"><h4><a href="https://www.easycalculation.com/maths-dictionary/convex_set.html">What is convex set - Definition and Meaning</a></h4><p>A convex set is defined as the region, in which any two points lies within the region, while the points on the line segment which connect these points also lies within the region. If point lies outside the region, then it is a non convex set. Formula : A cube is a convex set.</p></blockquote>
<script async src="//cdn.embedly.com/widgets/platform.js" charset="UTF-8"></script>



<blockquote class="embedly-card"><h4><a href="https://mathworld.wolfram.com/Convex.html">Convex</a></h4><p>A set in Euclidean space is convex set if it contains all the line segments connecting any pair of its points. If the set does not contain all the line segments, it is called concave. A convex set is always star convex, implying pathwise-connected, which in turn implies connected.</p></blockquote>
<script async src="//cdn.embedly.com/widgets/platform.js" charset="UTF-8"></script>



### Convex and Concave Functions

**Convex Function** are functions such that when you join two points to make a straight line and if the **line lies above** the function curve, then it is called as **convex function**. 

In simple terms, the convex funtions refers to a function that is in the shape of cup $$\cup$$ or **bowl like** object. Imagine, pouring water into to it to fill it up. The resulting **filled-in** set is called **epigraph** of the convex function, is a **convex set**.

If the **line lies below** the function curve then it is called as **concave function**. In simple terms,  **concave functions** refer to a function that is in the shape of inverted cup $$\cap$$. **Note:** The concave function is **negative of convex function**.



### Convex Optimization

In Summary, the constrained optimization problem is called as **convex optimization** if

$$ minimize_x{ f(x)} \\  subject\  to \ \  g_i(x) <= 0 for\ all \ i \\ and \ h_j(x) = 0 \ for\ all \ j$$

where **f(x) and g(x) are convex functions** and all $$h_j(x)$$ = 0 are **convex sets**.

Two ways to solve constrained optimization problem and they are:

* **Linear Programming**
* **Quadractic Programming**

## What is Linear Programming?

Linear programming uses **mathematical model** to describe the problem of concern.

The adjective **linear** means that **all mathematical functions** in this model are required to be **linear functions**.

The word **programming** DOES NOT refer to computer programming, rather it is synonym for **planning**.

Thus, it is a **planning of activities** to obtain the **optimal** result among all **feasible solutions**.

An **efficient solution procedure** is available called **simplex method** to solve linear programming problem.

---



## References

* [Convex Function](https://en.wikipedia.org/wiki/Convex_function)

* [ Operations Research - Youtube Channel Yong Wang](https://www.youtube.com/playlist?list=PLgA4wLGrqI-ll9OSJmR5nU4lV4_aNTgKx)

* Convex optimization

* Differentiability of objective function.

  