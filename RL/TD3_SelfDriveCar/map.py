# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

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

## car game - do refactoring later
#from game import Game

# Importing the Dqn object from our AI in ai.py
#from ai import Dqn
from td3_ai import ReplayBuffer, TD3, Actor
from torchsummary import summary
from gym import spaces

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1429')
Config.set('graphics', 'height', '660')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
#brain = Dqn(5,3,0.9)


brain = 0
action2rotation = [0,5,-5]
last_reward = 0
scores = []
im = CoreImage("./images/MASK1.png")

# textureMask = CoreImage(source="./kivytest/simplemask1.png")

#--------------------------- Globals -------------------------------------
# Initializing the map
first_update = True

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



#----------------------------------------------------------------
def init_map():
    global sand
    global goal_x
    global goal_y
    global first_update
    global swap
    global mask_img
    global longueur
    global largeur

    print('initializing sand...')
    sand = np.zeros((longueur,largeur))
    img = PILImage.open("./images/mask.png").convert('L')
    mask_img = np.asarray(img)
    sand = np.asarray(img)/255
    goal_x = TARGET_LOC_X # target x
    goal_y = TARGET_LOC_Y  # target y
    first_update = False
    swap = 0

#----------------------------------------------------------------
# evaluate_policy
#----------------------------------------------------------------
def evaluate_policy(policy, eval_episodes=10):
  avg_reward = 0.
  game = Game()

  for _ in range(eval_episodes):
    obs = game.reset()
    done = False
    while not done:
      action = policy.select_action(np.array(obs))
      obs, reward, done, _ = env.step(action)
      avg_reward += reward

  avg_reward /= eval_episodes

  print("---------------------------------------")
  print("Average Reward over the Evaluation Step: %f" % (avg_reward))
  print("---------------------------------------")
  return avg_reward

# Initializing the last distance
last_distance = 0
#--------------------------- Training Related ---------------------------

#### ------ Parameters -------------
start_timesteps = 50
max_episode_steps = 100
timesteps_since_eval = 0
# How often the evaluation step is performed (after how many timesteps)
eval_freq = 5e3
max_timesteps = 5e5  # Total number of iterations/timesteps

save_models = True  # Boolean checker whether or not to save the pre-trained model
expl_noise = 0.1  # Exploration noise - STD value of exploration Gaussian noise
batch_size = 100  # Size of the batch
# Discount factor gamma, used in the calculation of the total discounted reward
discount = 0.99
tau = 0.005  # Target network update rate
# STD of Gaussian noise added to the actions for the exploration purposes
policy_noise = 0.2
# Maximum value of the Gaussian noise added to the actions (policy)
noise_clip = 0.5
# Number of iterations to wait before the policy network (Actor model) is updated
policy_freq = 2


# cyclic replay buffer
replay_buffer = ReplayBuffer(100)
state_dim = (IMAGE_PATCH_WIDTH, IMAGE_PATCH_HEIGHT)
action_dim = 3
max_action = 1.0

actor = Actor(state_dim, action_dim, max_action)
summary(actor, input_size=(1,40,40))

# build network - TD DDPG
#policy = TD3(state_dim, action_dim, max_action)

# evaluation of the model
#evaluations = [evaluate_policy(policy)]

#--------------------------- Game -------------------------------------
class Game(Widget):

    car = ObjectProperty(None)

    # def __init__(self):
    #     #self.car = ObjectProperty(None)
    #     self.reward = 0.0
    #     self.prev_reward = 0.0
    #     ## actions - steer, gas, brake
    #     ## steer: range -1 to 1 => -1 left, 0 straight, > 0 right
    #     self.action_space = spaces.Box(
    #         np.array([-1, 0, 0]), np.array([+1, +1, +1]), dtype=np.float32)
    #     self.observation_space = spaces.Box(
    #         low=0, high=255, shape=(IMAGE_PATCH_HEIGHT, IMAGE_PATCH_WIDTH), dtype=np.uint8)
    #     #self.car.center = self.center

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

    # get the image patch of certain size
    def get_sand_image_patch(self, width, height):
        global sand
        global mask_img

        x = int(self.car.x)
        y = int(self.car.y) - (height // 2)
        sand_patch = sand[x:x+width, y:y+height]
        return sand_patch

    def get_image_patch(self, width, height):
        x = int(self.car.x)
        y = int(self.car.y) - (height // 2)
        img_patch = mask_img[x:x+width, y:y+height]
        return img_patch
    
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

        new_obs = self.get_sand_image_patch(IMAGE_PATCH_WIDTH, IMAGE_PATCH_HEIGHT)
         
        #reward
        step_reward = 0
        done = False

        if action is not None: # step without action, called from reset()
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
        #print('replay_buffer size:', replay_buffer.max_size)
        # for i in range(5):
        #     self.step(1)
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

            # find action
            # for intial timesteps, sample the actions.
            if total_timesteps < start_timesteps:
                action = self.action_space.sample()

            

            # get the next state and reward
            new_obs, reward, done, info = self.step(action)

            # We check if the episode is done
            done_bool = 0 if episode_timesteps + 1 == max_episode_steps else float(done)

            # We increase the total reward
            episode_reward += reward
  
            # We store the new transition into the Experience Replay memory (ReplayBuffer)
            replay_buffer.add((obs, new_obs, action, reward, done_bool))

            # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
            obs = new_obs
            episode_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1
             


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

    def move(self):
        #print('pos:', self.pos)
        self.pos = Vector(*self.velocity) + self.pos
        #self.rotation = rotation
        #self.angle = self.angle + self.rotation

 

#----------------------------------------------------------------
# Adding the painting tools
class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            print('on_touch_down')
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1
            img = PILImage.fromarray(sand.astype("uint8")*255)
            img.save("./images/sand.jpg")

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        print('on_touch_move')
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1

            
            last_x = x
            last_y = y


##----------------------------------------------------------------





#----------------------------------------------------------------
# Adding the API Buttons (clear, save and load)
# The name of the .kv file should match with the name of this App.
#
class CarApp(App):

    def build(self):
        print('Build the Game')
        parent = Game()
        parent.serve_car()

        print('Init car location')
        #parent.reset()

        print('Init the map')
        init_map()

        print('Painting the canvas')
        self.painter = MyPaintWidget()

        Clock.schedule_once(parent.train)
         


        ###-- buttons ---##
        # clearbtn = Button(text = 'clear')
        # savebtn = Button(text = 'save', pos = (parent.width, 0))
        # loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        # clearbtn.bind(on_release = self.clear_canvas)
        # savebtn.bind(on_release = self.save)
        # loadbtn.bind(on_release = self.load)
        # parent.add_widget(self.painter)
        # parent.add_widget(clearbtn)
        # parent.add_widget(savebtn)
        # parent.add_widget(loadbtn)
        return parent


 

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))
        
    def save(self, obj):
        print("saving brain...")
        #brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        #brain.load()

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
