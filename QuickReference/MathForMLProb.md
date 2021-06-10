# Math For ML - Probability and Statistics

[TOC]

---

### What is Probability?

Probability is study about **randomness**. 

The *probability* is a number which defines the **chance** or the **likelihood** of a particular event's occurrence.

It's represented as *proportion* or *percentage*. In the case of **proportion** (comparison between two numbers), **it ranges from 0 to 1**. In the case of percentage, it ranges from 0% to 100%.

---

###What is Statistics?

Statistics is all about finding the **basic number(s)** from data. 

The **basic number(s)** are used to describe the data. The following are some of the **basic number(s) or parameters** :

 *   **Mean** (average value)
 *   **Median ** (middle value of data)
 *   **Mode** (often occurring value, high frequence value)
 *   **Variance** (spread of data from mean) etc.

---

### Two Schools of Probability

* Based on frequency (Frequentist Probability).
  * Events are observed and counted. The probability is based on the frequency of occurrences. Hence **frequentist** probability. **It is objective**.
* Bayesian Probability
  * **It's subjective**. Probabilities are assigned based on **evidence and belief** and centered around *Bayes theorem*. 
  * This allows to assign probability for the event so far **not observed**.

---

### What is a random variable?

A random variable is a function which maps the **event (state space) to number of occurrences**, which in turn helps to calculate the probability.

---

### Probability Distribution

It's a summary of **probabilities for the possible values** of a random variable.

The following are the **properties** or **moments** of the distribution.

 * Mean (1st moment) - average value.
 * Variance (2nd moment) - spread of the values from mean
 * Skewness (3rd moment) - Postive / Negative skewed.
 * Kurtosis (4th moment) - Measure of tailed ness.

Broadly the Probability Distributions are classified as

* **Discrete Probability Distribution**
* **Continuous Probability Distribution**

---

#### Discrete Probability Distribution

*Discrete - Finite set of states*

* Bernoulli / Binomial Distribution
  * **Bernoulli distribution** : Just one trial with one outcome out of possible two.
    * A single coin flip
  * **Binomial Distribution**: Multiple independent Bernoulli trials.
    * A sequence of multiple independent coin flips.
* **Multinomial Distribution**
* **Possion Distribution** [*For Possion Distribution - Mean and Variance are same*]



Uses in ML : Binary Classification, Multi class classification.

The **Probability Mass Function (PMF)** returns the probability of a given outcome.

---

#### Continuous Probability Distribution

*Continuous - Value from a range of real valued numerical values*.

The relationship between the events of a continuous random variable and their probabilties is called Continuous Probability Distribution . It's summarized by *Probability Density Function*.

- Parametric Distribution

  The following are the **parameters** present in a continuous distribution (**one or more but not all parameters**). 

  - Scale : Defines the **range** of a distribution.
  - Shape : Defines the **shape** of a distribution.
  - Location : Locates / **shifts** the distribution on horizontal axis.

  The following are the popular continuous probability distributions.

  - **Exponential Distribution** : Only **Scale** parameter.
  - **Normal / Gaussian ** Distribution: $N(\mu,\sigma^2)$  , $\mu$ is the location parameter and $\sigma$ is the scale parameter. This is also called as **Bell** curve.
  - **Standard Normal Distribution** : $\mu$ = 0 and $\sigma$ = 1. This is also called as **standarization**.
  - Pareto Distribution - (*80/20*)

- Non-Parametric Distribution

  The following are some **non-parameteric** distribution. They are formed on the basis of **normal distribution**.

  * **Chi-squared Distribution** 

    * Based on Std. Normal Distribution. Can be thought of as "square" of a selection taken from std. normal distribution.

    * It is represented as $ Q = \Sigma_{i=1}^{k}Z_i^2  =  \chi_k^2 $

    * It is represented as $\chi^2(k)$, where **k is the degrees of freedom**.

    * Let $Z_1$ be the Std. Normal Distribution i.e., $Z_1 = \frac{X_1 - \mu}{\sigma}$

      $\chi^2(1) = Z_1^2 = (\frac{X_1 - \mu}{\sigma})^2​$

      In the above case, the degrees of freedom is 1.

      

  * **Student t Distribution** or **t-distribution**

    $t \ distribution =  \frac{ std. \ normal \ distribution}{\chi^2 \  distribution}$

  * **F Distribution**

    It's a ratio of two $\chi^2$ distributions.

---



### Sampling and Estimation

It is a process of selecting  a subset from population to make inference. By using the sample, the population parameters like Mean, Variance and Proportions of data can be estimated.

They can be classified as 

* Non-Probabilistic Sampling

* Probabilistic Sampling

  * Random Sampling

    * **With Replacement**

      Once the item is selected to the sample, that item **will be** removed from the population.

    * **WithOut Replacement**

      Once the item is selected to the sample, that item **will not be** removed from the population. Hence, sample items likely to repeat.

  * Stratified

  * Bagging ( Boostrap Aggregating)

  * Boosting 

---

#### Central Limit Theorem

In Simple terms, **irrespective of the population distribution**, central limit theorem states that for a **large sample** drawn from a population, the **sample mean and sample standard deviation follows the population**.



Note that the important assumption is that the **population** is identically distributed. The Central Limit Theorem forms the basis for hypothesis tests such as Z-test and t-test.

---

#### Confidence Interval

We **estimate population parameters** such as mean, variance, proportion and propbability distribution parameters - scale, shape and location from  a **sample** using techniques such as **moments** and **maximum likelihood estimation (MLE)**. 

Instead of predicting a single value, the **confidence interval** is a range in which the value of a population parameter is likely to lie with **certain probability**.

---

### Hypothesis Testing

Hypothesis is a claim made by a person / organization. The claim is usually about the **population parameters - mean, proportions etc**. We seek **evidence** from a sample for the support of the claim.

Hypothesis testing consists of **two complementary** statements called **null hypothesis** and **alternative hypothesis**. It is used for checking the validity of the claim using evidence found in a sample data.

Hypothesis testing is a process used for **either rejecting or retaining a null hypothesis**.

#### Kinds of Tests

Broadly they are classified as:

* Parametric Tests

  When the some of the population parameters are known, use sample to find evidence to support null hypothesis.

  * When **Population mean $\mu​$ and population std dev are known** $\sigma​$ , use Z-statistic test.

    $Z\ = \frac{\bar{X} -\mu}{\sigma /\sqrt(n)}$

    Where $\bar{X}$ is the sample mean and *n* is the number of samples.

  * When **population mean $\mu$ is known and population std dev $\sigma$ is unknown**, use t-statistic test.

    $t\ statistic = \frac{\bar{X} - \mu}{S/ \sqrt(n)}$

    Where S is the sample std. deviation and n is the number of samples.

* Non Parametric Tests

  

A Type I error occurs when you **reject** a null hypothesis that in fact is **TRUE**. This is also called as **level of significance** represented as **$\alpha$. ** Usually it's 0.05 or 0.1.

A Type II error occurs when you **do not reject** a null hypothesis that is in fact **FALSE**. It's represented as **$\beta$**.

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

  





 