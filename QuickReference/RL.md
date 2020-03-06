# Reinforcement Learning

---

[TOC]

----

## Reference Materials

* [Berkeley AI Course Videos](<http://ai.berkeley.edu/lecture_videos.htm>)
* Book: Reinforcement Learning by Sutton
* 

---



**Learning from interaction** is the fundamental idea underlying the learning and intelligence. The **Reinforment Learning** is inspired by **biological learning systems**.

**Reinforcement Learning (RL)** is a learning what to do - **how to map the situations to actions**- so to maximize the numerical reward. 

The following are the different elements of a reinforcement learning alogrithm:

* agent

* environment

  

Note that for the RL, there is **no traning set**. The agent has to taken action based on the situation (**environment**) it presents and it has impact on when does the agent gets the **reward**.

One of the **main challenge** of the RL problem is : **exploration** vs. **exploitation**.

The agent has to **exploit** the action which it has already experienced / learned to get the **reward** but it also has to **explore** to select better action in the future.

## Elements of Reinforcement Learning

Apart from **agent and environment**, the following are **sub elements** of reinforcement learning.

* *reward signal*
* *policy*
* *value function*
* *model (optional)*

The **reward signal** defines the **goal of the reinforcement learning.**

The **policy** defines **the agent's reponse to a situation**. The **policy** can be implemented as a **simple function or a lookup table or in some cases the search process.** The policy may be **stochastic (ie., random)**, specifying the **probabilities** for each action. 

The **value function** defines the **what is good for long-term**, ie., the **total amount** of reward the agent can expect to accumulate over the future starting from that state.

The **Rewards** are provided by the environment. The **values** are **estimated and re-estimated** from the sequences of observations an agent makes over its lifetime.

The **model** mimics the **behavior of environment**. It allows **inferences** to be made about how the environment will behave. For example, **given a state and action**, the model might predict the **next resultant state and reward**. **Models** are used for *planning*, helps in deciding the next course of action by considering the possible future situations before they are experienced.

* *Model-based* : 
* *Model-free*: Also called as trial-and-error learners.

 

## Sequential Decision Making Problems

Sequential Decision Making problems are kind of problems where the earlier decisions **influcences** the later **available choices**.

### Framework to Solve

The **mathematical framework** to solve **Sequential Decision Making problems** is **Markovian Decision Process (MDP)**.

In the **Markovian environment**, **the state *t* depends on events only at t-1 and NOT about the history of how you reached state t-1.**

In other words, the **Markov Property** depends only on the **current** state and **not on previous history **(how that state was reached).

The transition from one state to another can be:

* Deterministic (known in advance)
* Stochastic (ie., random with probabilities).

---

Let's focus on **finite** MDP, where the **set of states, actions and rewards all have a finite number of elements**.

The **key elements** of solving MDP are:

* returns
* value functions
  * *policy*
* Bellman equation

The **returns** is a function of future rewards which the agent tries to maximize.

The **value functions** assigned to each **state or state-action pair** is the **expected return** from that state.

The **Bellman equation** helps in determining the optimal **value functions**. The **Bellman equation** *expresses the relationship between the current state and its successor states.*

All **Reinforcement Learning algorithms** involve estimating **value functions** - functions of states (or state-action pairs) that estimates **how good** it is for the agent to be in a given state (or how good to perform a given action in a given state). The **value functions** are defined with respect to particular ways of acting, called **policies**. The **policy** is *mapping of states to probabilities of selecting each possible action*.

---

## How to compute the optimal policy?

The **reinforcement learning** uses the *value functions* to organise and structure the search for **good policies.** The value functions can be computed by the following approaches:

* Dynamic Programming (DP)
  * Complete knowledge of the environment should be known.
  * Not practical for large problems.
  * *Curse of Dimensionality* - the number of states often grows exponentially with the number of state variables.
* Monte Carlo Methods
  * No need to know the complete knowledge of the environment.
  * Based on *averaging* sample returns.
* Temporal-Difference Learning (TD)
  * Is a combination of Dynamic Programming and Monte Carlo Methods.
  * Like Monte Carlo Method, TD can learn directly from the raw experience without a model of the environment's dynamics. So, TD is a **model-free** learning algorithm.That is, the agent does not known about the reward and transition systems (tables).
  * Like DP, TD methods update estimates.

---

## Temporal Difference (TD) Learning

Temporal Difference algorithms enables the agent to learn through each action it takes. TD updates the knowledge of the agent on every action instead of updating after reaching the end state / goal.

The **main question** is **how does the agent chooses action?** ie., **what action the agent will take in a particular state (policy)?**

There are **two control policies** which will help the agent to take the next action. They are

* SARSA (State-Action-Reward-State-Action)
* Q-Learning

