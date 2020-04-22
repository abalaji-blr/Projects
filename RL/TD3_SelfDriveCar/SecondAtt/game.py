# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time
from gym import spaces

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from PIL import Image as PILImage
from kivy.graphics.texture import Texture

## configurations
IMAGE_PATCH_WIDTH = 40
IMAGE_PATCH_HEIGHT = 40

START_LOC_X = 600
START_LOC_Y = 350

TARGET_LOC_X = 920
TARGET_LOC_Y = 270

PLAYFIELD_X = 1429
PLAYFIELD_Y = 660

longueur = 0
largeur = 0
goal_x = 0
goal_y = 0

sand = np.zeros((1469, 660))

#----------------------------------------------------------------
# Creating the car class


class Car(Widget):

    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)
    x = NumericProperty(0)
    y = NumericProperty(0)

    def move(self):
        #print('pos:', self.pos)
        self.pos = Vector(*self.velocity) + self.pos
        #self.rotation = rotation
        #self.angle = self.angle + self.rotation

#--------------------------- Game -------------------------------------
class Game(Widget):

    car = ObjectProperty(None)
    #car = Car()

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

        self.car.x = 600
        self.car.y = 350
        #print('car location:', self.car.center)

        self.reward = 0.0
        self.prev_reward = 0.0
        ## actions - steer, gas, brake
        ## steer: range -1 to 1 => -1 left, 0 straight, > 0 right
        self.action_space = spaces.Box(
            np.array([-1, 0, 0]), np.array([+1, +1, +1]), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(IMAGE_PATCH_HEIGHT, IMAGE_PATCH_WIDTH), dtype=np.uint8)

    def set_car_location(self, x, y):
        self.car.x = x
        self.car.y = y

    def print_car_location(self):
        print('car location: ', self.car.x, ' ', self.car.y)

    ##---------- reset ----------------
    def reset(self):
        global longueur
        global largeur

        #self.car.center = self.center
        #self.car.center  = (600,310)
        # set car to starting location.
        self.car.x = 600
        self.car.y = 350
        longueur = self.width
        largeur = self.height

        self.reward = 0.0
        self.prev_reward = 0.0

        # use step() and obtain the state / observation
        obs = self.step(None)[0]
        #obs = self.get_sand_image_patch(IMAGE_PATCH_WIDTH, IMAGE_PATCH_HEIGHT)
        return(obs)

    #------------------- step ----------------
    # input: action
    # outputs: new_obs, reward, done-flag, debug-info
    def step(self, action):
        global goal_x
        global goal_y

        print('car location: ', self.car.x, self.car.y)
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx, yy))/180.

        if action is not None:
            self.car.move()

        new_obs = self.get_sand_image_patch(
            IMAGE_PATCH_WIDTH, IMAGE_PATCH_HEIGHT)

        #reward
        step_reward = 0
        done = False

        if action is not None:  # step without action, called from reset()
            # calculate reward
            self.reward -= 0.1
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            #
            # update done flag
            # set done to True when it hits the wall/ hits the boundary.
            x, y = self.car.x, self.car.y
            if abs(x) > PLAYFIELD_X and abs(y) > PLAYFIELD_Y:
                done = True
                step_reward = -100

        return new_obs, step_reward, done, {}
    # get the image patch of certain size

    def get_sand_image_patch(self, width, height):
        global sand
        global mask_img

        x = int(self.car.x)
        y = int(self.car.y) - (height // 2)

        # in the boundaries, we may not be able to get 
        # exact (width, height). So, pad them with zeros.
        result = np.zeros([1,width, height])

        sand_patch = sand[x:x+width, y:y+height]
        #sand_patch = np.expand_dims(sand_patch, axis=0)
        result[0, :sand_patch.shape[0], :sand_patch.shape[1]] = sand_patch
        return result

    def get_image_patch(self, width, height):
        x = int(self.car.x)
        y = int(self.car.y) - (height // 2)
        img_patch = mask_img[x:x+width, y:y+height]
        return img_patch

    #-------------------------- Training --------------------------

    def train(self, dt):
        global done
        global total_timesteps, start_timesteps
        global replay_buffer
        total_timesteps = 0
        timesteps_since_eval = 0
        episode_num = 0
        done = True
        t0 = time.time()

        print('Training...')
        max_timesteps = 10

        # We start the main loop over max timesteps
        while total_timesteps < max_timesteps:

            # If the episode is done
            if done:
                obs = self.reset()

                # set done to False
                done = False

                # Set rewards and episode timesteps to zero
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # # find action
            # # for intial timesteps, sample the actions.
            # if total_timesteps < start_timesteps:
            #     action = self.action_space.sample()

            # # get the next state and reward
            new_obs, reward, done, info = self.step(0)

            # # We check if the episode is done
            # done_bool = 0 if episode_timesteps + \
            #     1 == max_episode_steps else float(done)

            # # We increase the total reward
            # episode_reward += reward

            # # We store the new transition into the Experience Replay memory (ReplayBuffer)
            # replay_buffer.add((obs, new_obs, action, reward, done_bool))

            # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
            obs = new_obs
            episode_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1


