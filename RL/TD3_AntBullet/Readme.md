# TD DDPG (TD3) using AntBulletEnv

* [TD3 with AntBulletEnv](https://youtu.be/Ijp20J88S0k)

* <iframe width="560" height="315" src="https://www.youtube.com/embed/Ijp20J88S0k" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

  

* Env Info.

  ~~~
  env.observation_space: (28,)
  env.action_space: (8,)
  state_dim 28
  action_dim: 8
  max_action: 1.0
  ~~~

  

  ~~~
  print(env.action_space.high)
  print(env.action_space.low)
  
  [1. 1. 1. 1. 1. 1. 1. 1.]
  [-1. -1. -1. -1. -1. -1. -1. -1.]
  ~~~



~~~
env.reward_range

(-inf, inf)
~~~







