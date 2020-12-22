# Math For ML - Probability

[TOC]

## Probability

### Two Schools of Probability

* Based on frequency (Frequentist Probability).
  * Events are observed and counted. The probability is based on the frequency of occurrences. Hence **frequentist** probability. **It is objective**.
* Bayesian Probability
  * **It's subjective**. Probabilities are assigned based on **evidence and belief** and centered around *Bayes theorem*. 
  * This allows to assign probability for the event so far **not observed**.

### What is a random variable?

A random variable is a function which maps the **event (state space) to number of occurrences**, which in turn helps to calculate the probability.

---

### Probability Distribution

It's a summary of probabilities for the possible values of a random variable.

The following are the *properties* or *moments* of the distribution.

 * Mean (1st moment) - average value.
 * Variance (2nd moment) - spread of the values from mean
 * Skewness (3rd moment) - Postive / Negative skewed.
 * Kurtosis (4th moment) - Measure of tailed ness.

---

#### Discrete Probability Distribution

*Discrete - Finite set of states*

* Bernoulli / Binomial Distribution
  * Bernoulli distribution : Just one trial with one outcome out of possible two.
    * A single coin flip
  * Binomial Distribution: Multiple independent Bernoulli trials.
    * A sequence of multiple independent coin flips.
* Multinomial Distribution
* Possion Distribution [*For Possion Distribution - Mean and Variance are same*]



Uses in ML : Binary Classification, Multi class classification.

The **Probability Mass Function (PMF)** returns the probability of a given outcome.

---

#### Continuous Probability Distribution

*Continuous - Value from a range of real valued numerical values*.

The relationship between the events of a continuous random variable and their probabilties is called Continuous Probability Distribution . It's summarized by *Probability Density Function*.

* Normal or Gaussian Distribution
* Exponential Distribution
* Pareto Distribution - (*80/20*)

---

### Probability of Multiple Random Variables

Let's assume the random variables are X and Y and the event A is for X and event B for Y.

* Joint Probability
  * Probability of **two (or more) events occuring simultanously**.
  * $P(A \and B) = P(A \| B) * P(B)​$
  * $P(A\and B)$ is the **joint probability of events A and B**.
  * $P(A\| B)$ is the **conditional probability** 
  * It's also called as  **product rule** of probability.
* Marginal Probability
  * Probability of an **event (of a random variable)** irrespective of the outcomes of **other variables**.
  * It's **sum (or union)** over all probabilities of all events of the second variable for a **given fixed event** for a first variable.
  * It's also called as **sum rule** of probability.
* Conditional Probability
  * Probability of an **event** occurring in the presence of **one or more other events**.
  * $P(A\| B) = P(A \and B) \div P(B) \\  => P(A\|B) = \frac{ P(B\|A) * P(A) }{ P(B)}$
  * This is the **Bayes Theorem**.
  * The **joint distribution** can be **factorized (or decomposed)** into **conditional probability**.

---

### Probability Density Estimation

Given a *random sample of a variable*, we are trying to identify the *probability distribution for a random variable*. **Note that often times, we may not know the PDF for a random variable and we may not have all the possible outcomes**.

Following are some of the ways to identify the PDF for a random variable.

 * Plot the random samples in a histogram and identify the distribution.

 * Parameteric Density Estimation

    * It is a iterative (or trial and error process)
       * Identify the mean and variance from the samples (sample-mean and sample-variance).
       * Review the estimated PDF with data.
       * Transform the data to better fit the distribution.

 * Non-parametric Density Estimation

    * Kernel Smoothing

 * Maximum Likelihood Estimation (MLE), a **frequentist method**

    * Approach as a optimization or search problem.

    * Given a dataset X, we need to find the parameters $\theta$ , which fits the data.

    * $$
      P(X; \theta) - \theta  \; is \; probability \; distribution \; parameters \\
      => L (X; \theta) - L  \;is \;  the \; likelihood \; function \\
      => max L(X; \theta) - need \; to \; maximize \; \\
      => max L(x1, x2, ...; \theta) - joint \; probability \; distribution \; of \; samples for\; given \; \theta \\
      => \prod_{i=1}^{n} P(x_i; \theta) - joint \; probability \; can \; be \; stated \; as \; conditional \; probability \; for \; given \; \theta \\
      => \sum_{i=1}^{n} log P(x_i; \theta) - convert \; to \; log \; probabilities \\
      => min - \sum_{i=1}^{n} log P(x_i; \theta) - in \; optimization \; we \; need \; to \; minimize,\; so \; negative \\
      $$

   * MLE is used when there is **no prior probability info. is available**.

 * Maximum a Posteriori (MAP), a **Bayesian method**.

    * Let's assume **A** and **B** are the events from two **independent random variables X and Y**.

    * Find out the **conditional probability of A given one or more other events**, P(A\|B).

    * Conditional Probability can be calculated from the **joint probability**.

    * $P(A\|B)  = P(A \and B) \div P(B)$

    * At times, the **joint probability** is not easy to calculate.

    * Bayes Theorem provides a way to calculate **without the joint probability**.

    * 

    * $$
      P(A\|B) = \frac{P(B\|A) * P(A) } { P(B)}\\
      P(A\|B) \;is\; the\; poterior\; probability \\
      P(B\|A)\; is \;called\; the\; likelihood \\
      P(A) \;is \;the \;prior \\
      P(B) \; is \; the \; evidence
      $$

   * **MAP ** is appropriate when we have the **prior probability**.

---

### Probability Graphical Models

The probabilistic models can define the **relationships between the random variables** and can calculate probabilities. They are represented as **nodes and edges**. 

Broadly, there are two categories, depending on 

 1) how they encode the **independence among random variables ** and

 2) how they **factorize** the distribution.



* One represents dependency with **directed edges**  (aka **Bayesian Network**)

  * Directed graph

    * The **directed edges** represent the **conditional distribution**.

    * They represent **causal relationship**.

    * No Cycles - Directed Acyclic Graph (DAG) 

      * These are Bayesian Network.
      * They capture both conditionally dependent and conditionally independent relationships between the variables.

    * WIth Cycles

      * These are called as **Hidden Markov Model**. Used in **Reinforcement Learning**.

        * Used to model sequence of observations (think of **state machines** ).

        * The name **HMM** is due to the following two properties.

          * Hidden :  For the observer, the process whose state is **hidden** which generates certain output.

          * Markov Property: The **current state** is just dependent on the **input state** of the process and **does not depend on prior staes ( how the input state was arrived)**.

            

* Other represents the dependency with **undirected edges**  (aka **Markovian Random Field**)

  * Undirected graph
    * Can have cycles.
    * The **edges** represent the **joint probability**.

---

### Probabilistic Modelling

* Treat all the random variables as **conditionally independent**. - **Naive Bayes** classification algorithm.
* Fully conditionally dependent model.
* Hybrid model - some are conditionally **independent** and others are conditionally **dependent**. - **Bayesian Belief Network**.

---

### References

[Types of Distribution in Statistics](https://www.reddit.com/r/mathrock/comments/fhau9z/types_of_distribution_in_statistics/)

![Types of Distribution in Statistics](https://i.redd.it/ebs7blwt66m41.jpg>)



* [Continuous Statistical Distributions, SciPy. ](https://docs.scipy.org/doc/scipy/reference/tutorial/stats/continuous.html)

* [Probablisitic Graphical Models ](<https://towardsdatascience.com/introduction-to-probabilistic-graphical-models-b8e0bf459812>)

  





 