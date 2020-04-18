# SelfDriveCar using TD3 (TD DDPG)

* Configurations

  * Use image patch of size 40x40 for the next state.
  * Number of actions: (3)

* Actor Model

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

  * ~~~
    * inputs are : state (image patch of 1x40x40) and actions (3).
    * output: Q-value
    * Issues:
    	* How to concatenate the follwing tensors?
    		(1,40,40) and (3)
    		Probably, pad the second tensor (3) with zeroes and create something like
    		1x40x80??? or 1x40x44??
    ~~~

    