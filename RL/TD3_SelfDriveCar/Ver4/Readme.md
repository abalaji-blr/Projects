# Self Drive Car Using TD3 ( TD DDPG)

* Configurations
  * Use image patch of size 40x40 for the next state.

  * Number of actions:  3 ( left, straight, right)

  * Video:

  * <iframe width="560" height="315" src="https://www.youtube.com/embed/Xg93-rIAu2M" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

  * Source Code: [map.py]('./map.py')

* Actor Model

  * Input: State

  * Output: Action

  * ~~~
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1           [-1, 32, 10, 10]             544
                Conv2d-2             [-1, 64, 5, 5]           8,256
                Conv2d-3              [-1, 3, 1, 1]           4,803
     AdaptiveAvgPool2d-4              [-1, 3, 1, 1]               0
    ================================================================
    Total params: 13,603
    Trainable params: 13,603
    Non-trainable params: 0
    ----------------------------------------------------------------
    ~~~

    

* Critic Model

  * Inputs: State & Action

    * **State is 40x40**. use Conv network and finally use 'global average pool' the features.
    * **Multiply those *features* with *action***

  * Output: Q-value

  * ~~~
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1           [-1, 32, 10, 10]             544
                Conv2d-2             [-1, 64, 5, 5]           8,256
                Conv2d-3              [-1, 3, 1, 1]           4,803
     AdaptiveAvgPool2d-4              [-1, 3, 1, 1]               0
                Conv2d-5           [-1, 32, 10, 10]             544
                Conv2d-6             [-1, 64, 5, 5]           8,256
                Conv2d-7              [-1, 3, 1, 1]           4,803
     AdaptiveAvgPool2d-8              [-1, 3, 1, 1]               0
    ================================================================
    Total params: 27,206
    Trainable params: 27,206
    Non-trainable params: 0
    ----------------------------------------------------------------
    ~~~

  

  * 

    