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

#sand = np.zeros((1469, 660))
im = CoreImage("./images/MASK1.png")
print('original mask:', im.size)

#----------------------------------------------------------------


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

    # action : 0 : left turn
    # action : 1 : straight
    # action : 2 : right turn
    def move(self, action):
        if action is not None:
            self.pos = Vector(*self.velocity) + self.pos

            action2rotation = [-5, 0, 5] # left, straight, right
            self.rotation = action2rotation[action]
            self.angle    = self.angle + self.rotation

        #print('pos:', self.pos)
         

#--------------------------- Game -------------------------------------
class Game(Widget):

    car = ObjectProperty(None)
    #car = Car()

    def init_game(self):
        self.sand = np.zeros((self.width, self.height))
        img = PILImage.open("./images/mask.png").convert('L')
        print('PIL mask: ', img.size)
        # 
        # note that PIL uses different co-ordinate system.
        self.mask_img = np.asarray(img) # original mask image using PIL.

        self.sand = np.asarray(img)/255 # normalized image

        # target location
        goal_x = TARGET_LOC_X  # target x
        goal_y = TARGET_LOC_Y  # target y

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

        self.car.x = 600
        self.car.y = 350
        #print('car location:', self.car.center)

        self.reward = 0.0
        self.prev_reward = 0.0
        # 0-left, 1 - straight, 2 right turns.
        self.action_space = 3
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(IMAGE_PATCH_HEIGHT, IMAGE_PATCH_WIDTH), dtype=np.uint8)

    def set_car_location(self, x, y):
        self.car.x = x
        self.car.y = y

    def print_car_location(self):
        print('car location: ', int(self.car.x), ' ', int(self.car.y))

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

    def sample_action(self):
        return np.random.randint(0, self.action_space)

    def isCarOffLimits(self):
        res = False
        x, y = int(self.car.x), int(self.car.y)
        if x >= PLAYFIELD_X or x < 0:
            res = True
        elif y >= PLAYFIELD_Y or y < 0:
            res = True
        return res

    def isTargetReached(self):
        res = False

        x, y = int(self.car.x), int(self.car.y)
        window_size = 20 # num pixels
        if x >= goal_x - window_size and x <= goal_x + window_size \
            and y >= goal_y - window_size and y <= goal_x + window_size:
            res = True
        return res

    #------------------- step ----------------
    # input: action
    # outputs: new_obs, reward, done-flag, debug-info
    def step(self, action):
        global goal_x
        global goal_y
        global im

        #print('action:', action, ' car-location:(', int(self.car.x),',', int(self.car.y), ')', )
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx, yy))/180.

        if action is not None:
            x, y = int(self.car.x), int(self.car.y)
            #isCarOffLimits = self.isCarOffLimits
            if self.isCarOffLimits() == False:
                if self.sand[x, y] > 0:
                    # off the road
                    self.car.velocity = Vector(5, 0).rotate(self.car.angle)
                else:
                    # on road
                    self.car.velocity = Vector(10, 0).rotate(self.car.angle)
                self.car.move(action)

        new_obs = self.get_sand_image_patch(
            IMAGE_PATCH_WIDTH, IMAGE_PATCH_HEIGHT)

        #reward
        #step_reward = 0
        reward = 0
        done = False

        # Handle the rewards...
        if action is not None:  # step without action, called from reset()
            # update done flag
            # set done to True when it hits the wall/ hits the boundary.
            x, y = int(self.car.x), int(self.car.y)
            # mask_pixel_value = im.read_pixel(x,y)
            # print('image pixel', mask_pixel_value)

            # car is out of bounds, penalize
            if self.isCarOffLimits() == True:
                #print('**** car off limits *****')
                done = True
                reward = -100
            elif self.isTargetReached() == True:
                # target
                done = True
                print('********* Target Reached REWARD: 1000 ************')
                reward = 1000
            else:
                # off-road (hitting the building)
                if self.sand[x,y] > 0:
                    reward = -1
            
        #print('action: ', action, 'car: (', self.car.x, ', ', self.car.y)
        #print(f'action: {action} car:({int(self.car.x)},{int(self.car.y)}) reward: {reward} done:{done}')
        return new_obs, reward, done, {}
    # get the image patch of certain size

    def get_sand_image_patch(self, width, height):
        #global sand
        #global mask_img

        x = int(self.car.x)
        y = int(self.car.y) - (height // 2)

        # in the boundaries, we may not be able to get 
        # exact (width, height). So, pad them with zeros.
        result = np.zeros([1,width, height])

        sand_patch = self.sand[x:x+width, y:y+height]
        #sand_patch = np.expand_dims(sand_patch, axis=0)
        result[0, :sand_patch.shape[0], :sand_patch.shape[1]] = sand_patch
        return result

    def get_image_patch(self, width, height):
        x = int(self.car.x)
        y = int(self.car.y) - (height // 2)
        img_patch = mask_img[x:x+width, y:y+height]
        return img_patch
