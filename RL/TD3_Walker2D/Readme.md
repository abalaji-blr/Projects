# TD DDPG using Walker2DBulletEnv

* <iframe width="560" height="315" src="https://www.youtube.com/embed/fGrJF2OMgog" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

  * Env Info

  * ~~~
    env.observation_space: (22,)
    env.action_space: (6,)
    state_dim 22
    action_dim: 6
    max_action: 1.0
    
    print(env.action_space.high)
    print(env.action_space.low)
    
    [1. 1. 1. 1. 1. 1.]
    [-1. -1. -1. -1. -1. -1.]
    
    env.reward_range
    (-inf, inf)
    ~~~

    