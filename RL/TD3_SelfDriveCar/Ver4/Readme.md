# Self Drive Car Using TD3 ( TD DDPG)

* Configurations
  * Use image patch of size 40x40 for the next state.
  * Number of actions:  3 ( left, straight, right)

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

    * State is 40x40. use Conv network and finally use 'global average pool' the features.
    * Multiply those *features* with *action*

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

* Training

  * ~~~
    ---------------------------------------
    Average Reward over the Evaluation Step: -180.500000
    ---------------------------------------
    Start Training...
    New training....
    Total Timesteps: 65 Episode Num: 1 Reward: -157
     /Users/abalaji/myData/RL/session10/att4/td3_ai.py:110: UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.
       x1 = F.softmax(self.gap1(x1))
     /Users/abalaji/myData/RL/session10/att4/td3_ai.py:118: UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.
       x2 = F.softmax(self.gap2(x2))
     /Users/abalaji/myData/RL/session10/att4/td3_ai.py:136: UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.
       x1 = F.softmax(self.gap1(x1))
    Total Timesteps: 176 Episode Num: 2 Reward: -200
    Total Timesteps: 244 Episode Num: 3 Reward: -158
    Total Timesteps: 300 Episode Num: 4 Reward: -137
    Total Timesteps: 372 Episode Num: 5 Reward: -164
    Total Timesteps: 437 Episode Num: 6 Reward: -156
    Total Timesteps: 490 Episode Num: 7 Reward: -134
    Total Timesteps: 568 Episode Num: 8 Reward: -162
    Total Timesteps: 727 Episode Num: 9 Reward: -224
    Total Timesteps: 819 Episode Num: 10 Reward: -163
    Total Timesteps: 899 Episode Num: 11 Reward: -160
    Total Timesteps: 960 Episode Num: 12 Reward: -156
    Total Timesteps: 1016 Episode Num: 13 Reward: -144
    Total Timesteps: 1076 Episode Num: 14 Reward: -155
    Total Timesteps: 1485 Episode Num: 15 Reward: -482
    Total Timesteps: 1719 Episode Num: 16 Reward: -311
    Total Timesteps: 1765 Episode Num: 17 Reward: -121
    Total Timesteps: 1896 Episode Num: 18 Reward: -214
    Total Timesteps: 1961 Episode Num: 19 Reward: -156
    Total Timesteps: 2021 Episode Num: 20 Reward: -155
    Total Timesteps: 2435 Episode Num: 21 Reward: -448
    Total Timesteps: 2587 Episode Num: 22 Reward: -239
    Total Timesteps: 2666 Episode Num: 23 Reward: -167
    Total Timesteps: 2776 Episode Num: 24 Reward: -191
    Total Timesteps: 2836 Episode Num: 25 Reward: -145
    Total Timesteps: 2942 Episode Num: 26 Reward: -193
    Total Timesteps: 2990 Episode Num: 27 Reward: -125
    Total Timesteps: 3577 Episode Num: 28 Reward: -608
    Total Timesteps: 3733 Episode Num: 29 Reward: -234
    Total Timesteps: 4097 Episode Num: 30 Reward: -410
    Total Timesteps: 4185 Episode Num: 31 Reward: -173
    Total Timesteps: 4227 Episode Num: 32 Reward: -121
    Total Timesteps: 4884 Episode Num: 33 Reward: -682
    Total Timesteps: 5096 Episode Num: 34 Reward: -285
    ---------------------------------------
    Average Reward over the Evaluation Step: -187.600000
    ---------------------------------------
    Total Timesteps: 5155 Episode Num: 35 Reward: -154
    Total Timesteps: 5304 Episode Num: 36 Reward: -203
    Total Timesteps: 5377 Episode Num: 37 Reward: -161
    Total Timesteps: 5498 Episode Num: 38 Reward: -199
    Total Timesteps: 5564 Episode Num: 39 Reward: -152
    Total Timesteps: 5731 Episode Num: 40 Reward: -253
    Total Timesteps: 5798 Episode Num: 41 Reward: -163
    Total Timesteps: 5869 Episode Num: 42 Reward: -162
    Total Timesteps: 6004 Episode Num: 43 Reward: -222
    Total Timesteps: 6073 Episode Num: 44 Reward: -158
    Total Timesteps: 6128 Episode Num: 45 Reward: -141
    Total Timesteps: 6185 Episode Num: 46 Reward: -151
    Total Timesteps: 6244 Episode Num: 47 Reward: -154
    Total Timesteps: 6304 Episode Num: 48 Reward: -155
    Total Timesteps: 6365 Episode Num: 49 Reward: -156
    Total Timesteps: 6426 Episode Num: 50 Reward: -156
    Total Timesteps: 6487 Episode Num: 51 Reward: -156
    Total Timesteps: 6548 Episode Num: 52 Reward: -156
    Total Timesteps: 6609 Episode Num: 53 Reward: -156
    Total Timesteps: 6670 Episode Num: 54 Reward: -156
    Total Timesteps: 6731 Episode Num: 55 Reward: -156
    Total Timesteps: 7156 Episode Num: 56 Reward: -473
    Total Timesteps: 7362 Episode Num: 57 Reward: -275
    Total Timesteps: 7510 Episode Num: 58 Reward: -220
    Total Timesteps: 7662 Episode Num: 59 Reward: -241
    Total Timesteps: 7763 Episode Num: 60 Reward: -190
    Total Timesteps: 7917 Episode Num: 61 Reward: -232
    Total Timesteps: 8260 Episode Num: 62 Reward: -405
    Total Timesteps: 8327 Episode Num: 63 Reward: -163
    Total Timesteps: 8450 Episode Num: 64 Reward: -213
    Total Timesteps: 8530 Episode Num: 65 Reward: -170
    Total Timesteps: 8680 Episode Num: 66 Reward: -228
    Total Timesteps: 8769 Episode Num: 67 Reward: -175
    Total Timesteps: 9411 Episode Num: 68 Reward: -641
    Total Timesteps: 9496 Episode Num: 69 Reward: -162
    Total Timesteps: 9895 Episode Num: 70 Reward: -436
    Total Timesteps: 9992 Episode Num: 71 Reward: -180
    Total Timesteps: 10342 Episode Num: 72 Reward: -406
    ~~~

    

