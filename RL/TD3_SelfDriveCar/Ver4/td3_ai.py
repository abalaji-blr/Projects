
#------------------------ imports -------------------------------
import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque


#import pybullet_envs
#from gym import wrappers

#------------------------ Replay Buffer  -------------------------------
class ReplayBuffer(object):

  def __init__(self, max_size=1e6):
    self.storage = []
    self.max_size = max_size
    self.ptr = 0

  def add(self, transition):
    if len(self.storage) == self.max_size:
      self.storage[int(self.ptr)] = transition
      self.ptr = (self.ptr + 1) % self.max_size
    else:
      self.storage.append(transition)

    # if len(self.storage) % 10 == 0:
    #   print('Replay buffer size: ', len(self.storage))

  def sample(self, batch_size):
    ind = np.random.randint(0, len(self.storage), size=batch_size)
    batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
    for i in ind: 
      state, next_state, action, reward, done = self.storage[i]
      batch_states.append(np.array(state, copy=False))
      batch_next_states.append(np.array(next_state, copy=False))
      batch_actions.append(np.array(action, copy=False))
      batch_rewards.append(np.array(reward, copy=False))
      batch_dones.append(np.array(done, copy=False))
    return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions),\
            np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)


#------------------------ Actor Model  -------------------------------
# input: state
# output: action
class Actor(nn.Module):
  def __init__(self, state_dim, action_dim, max_action):
    super(Actor, self).__init__()

    self.conv1 = nn.Conv2d(1, 32, 4, stride=4) # 1x40x40 => 10x10
    self.conv2 = nn.Conv2d(32, 64, 2, stride=2) # 10x10 => 5x5
    self.conv3 = nn.Conv2d(64, action_dim, 5, stride=1) # 5x5 => 3
    self.gap = nn.AdaptiveAvgPool2d((1,1)) # global ave pooling

    self.max_action = max_action

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = self.max_action * F.softmax(self.gap(x))
    return x

#------------------------ Twin Crtics -------------------------------

# input: State dim (1x40x40 - image patch) and action_dim - (3)
# output: q-value
#
# Issues:
# 1) How to concatenate (1,40,40) and (3) tensors?
#    Pad zeros to second tensor to make it to size (40) and merge 
#    to make it to (1,40,80)?? or any other better alternative?
#
#
class Critic(nn.Module):
  
  def __init__(self, state_dim, action_dim):
    super(Critic, self).__init__()
    # Defining the first Critic neural network
    self.conv1 = nn.Conv2d(1, 32, 4, stride=4)  # 1x40x40 => 10x10
    self.conv2 = nn.Conv2d(32, 64, 2, stride=2)  # 10x10 => 5x5
    self.conv3 = nn.Conv2d(64, action_dim, 5, stride=1)  # 5x5 => 3
    self.gap1 = nn.AdaptiveAvgPool2d((1, 1))  # global ave pooling

    # Defining the second Critic neural network
    self.conv4 = nn.Conv2d(1, 32, 4, stride=4)  # 1x40x40 => 10x10
    self.conv5 = nn.Conv2d(32, 64, 2, stride=2)  # 10x10 => 5x5
    self.conv6 = nn.Conv2d(64, action_dim, 5, stride=1)  # 5x5 => 3
    self.gap2 = nn.AdaptiveAvgPool2d((1, 1))  # global ave pooling


  def forward(self, x, u):
    #xu = torch.cat([x, u], 1)
    #xu = x
    # use x (image) and reduce them to n_classes
    # then, apply u actions on the generated output for NN to find out the Q-value
    #
    # Forward-Propagation on the first Critic Neural Network
    x1 = F.relu(self.conv1(x))
    x1 = F.relu(self.conv2(x1))
    x1 = F.relu(self.conv3(x1))
    x1 = F.softmax(self.gap1(x1))
    #x1 = torch.sum(x1 * u)
    x1 = x1 * u

    # Forward-Propagation on the second Critic Neural Network
    x2 = F.relu(self.conv4(x))
    x2 = F.relu(self.conv5(x2))
    x2 = F.relu(self.conv6(x2))
    x2 = F.softmax(self.gap2(x2))
    # print('x2:', x2)
    # print('x2 shape:', x2.size())
    # print('u:', u)
    # print('u size:', u.size())
    #mul = x2 * u 
    #print('mul shape:', mul.size())
    #print(mul)
    #x2 = torch.sum(x2 * u)
    x2 = x2 * u
    return x1, x2

  def Q1(self, x, u):
    #xu = torch.cat([x, u], 1)
    #xu = x
    x1 = F.relu(self.conv1(x))
    x1 = F.relu(self.conv2(x1))
    x1 = F.relu(self.conv3(x1))
    x1 = F.softmax(self.gap1(x1))
    #x1 = torch.sum(x1 * u)
    x1 = x1 * u
    #print('x1: ', x1)
    #print('u:', u)
    #print('Q1-value: ', x1)
    return x1


#------------------------ TD3 -------------------------------
# Selecting the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Building the whole Training Process into a class


class TD3(object):

  def __init__(self, state_dim, action_dim, max_action):
    self.actor = Actor(state_dim, action_dim, max_action).to(device)
    self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
    self.critic = Critic(state_dim, action_dim).to(device)
    self.critic_target = Critic(state_dim, action_dim).to(device)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
    self.max_action = max_action

  def select_action(self, state):
    state = torch.Tensor(np.expand_dims(state, axis=0)).to(device)
    #state = torch.Tensor(state).to(device)
    #print('state- TD3: ', state.shape)
    return self.actor(state).cpu().data.numpy().flatten()

  def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, \
            tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):

    for it in range(iterations):

      # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
      batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(
          batch_size)
      state = torch.Tensor(batch_states).to(device)
      next_state = torch.Tensor(batch_next_states).to(device)
      action = torch.Tensor(batch_actions).to(device)
      reward = torch.Tensor(batch_rewards).to(device)
      done = torch.Tensor(batch_dones).to(device)

      # Step 5: From the next state s’, the Actor target plays the next action a’
      next_action = self.actor_target(next_state)

      # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
      noise = torch.Tensor(batch_actions).data.normal_(
          0, policy_noise).to(device)
      noise = noise.clamp(-noise_clip, noise_clip)
      next_action = (
          next_action + noise).clamp(-self.max_action, self.max_action)

      # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
      target_Q1, target_Q2 = self.critic_target(next_state, next_action)

      # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
      target_Q = torch.min(target_Q1, target_Q2)

      # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
      target_Q = reward + ((1 - done) * discount * target_Q).detach()

      # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
      current_Q1, current_Q2 = self.critic(state, action)

      # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
      critic_loss = F.mse_loss(current_Q1, target_Q) + \
          F.mse_loss(current_Q2, target_Q)

      # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()

      # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
      if it % policy_freq == 0:
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
          target_param.data.copy_(
              tau * param.data + (1 - tau) * target_param.data)

        # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
          target_param.data.copy_(
              tau * param.data + (1 - tau) * target_param.data)

  # Making a save method to save a trained model
  def save(self, filename, directory):
    torch.save(self.actor.state_dict(), '%s/%s_actor.pth' %
               (directory, filename))
    torch.save(self.critic.state_dict(), '%s/%s_critic.pth' %
               (directory, filename))

  # Making a load method to load a pre-trained model
  def load(self, filename, directory):
    self.actor.load_state_dict(torch.load(
        '%s/%s_actor.pth' % (directory, filename)))
    self.critic.load_state_dict(torch.load(
        '%s/%s_critic.pth' % (directory, filename)))

#------------------------ imports -------------------------------
