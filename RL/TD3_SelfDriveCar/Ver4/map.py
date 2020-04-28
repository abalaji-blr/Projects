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
from td3_ai import ReplayBuffer, TD3, Actor, Critic
from torchsummary import summary
from gym import spaces
from game import Game

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
#action2rotation = [0,5,-5]
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



# Initializing the last distance
last_distance = 0
#--------------------------- Training Related ---------------------------

#### ------ Parameters -------------
start_timesteps = 1000  # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
#max_episode_steps = 10 # for debug
max_episode_steps = 1000  
timesteps_since_eval = 0
# How often the evaluation step is performed (after how many timesteps)
eval_freq = 5e3
#eval_freq = 2500

#max_timesteps = 5e5  # Total number of iterations/timesteps
max_timesteps = 50000  # Total number of iterations/timesteps

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
replay_buffer = ReplayBuffer(100000) # for debug 1e5
#replay_buffer = ReplayBuffer() # default 1 million
state_dim = (IMAGE_PATCH_WIDTH, IMAGE_PATCH_HEIGHT)
action_dim = 3
max_action = 1.0

actor = Actor(state_dim, action_dim, max_action)
summary(actor, input_size=(1,40,40))

critic = Critic(state_dim, action_dim)
summary(critic, [(1,40,40), (1,1,action_dim)]) # multiple inputs 
#summary(critic, [(1, 40, 40), (action_dim)])


# build network - TD DDPG
policy = TD3(state_dim, action_dim, max_action)


# set the environment
env = Game()

 
#----------------------------------------------------------------
# evaluate_policy
#----------------------------------------------------------------
def evaluate_policy(policy, eval_episodes=10):
  avg_reward = 0.
  global env
  #game = Game()

  for _ in range(eval_episodes):
    obs = env.reset()
    done = False
    while not done:
      action = policy.select_action(np.array(obs))
      #print('action from TD3:', np.argmax(action))
      action = np.argmax(action)
      obs, reward, done, _ = env.step(action)
      #print('action: ', action, ' reward: ', reward, ' done: ', done)
      avg_reward += reward

  avg_reward /= eval_episodes

  print("---------------------------------------")
  print("Average Reward over the Evaluation Step: %f" % (avg_reward))
  print("---------------------------------------")
  return avg_reward


# evaluation of the model
evaluations = []
#evaluations = [evaluate_policy(policy)]
 
def verify_train():
    print('Verify Traning Process...')
    for i in range(50):
        action = np.random.randint(0, 3)
        env.step(action)

def new_train(dt):
    print('New training....')
    global start_timesteps
    global replay_buffer

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    episode_reward = 0
    episode_timesteps = 0
    file_name = 'car_td3'
    done = True
    t0 = time.time()

    #max_timesteps = 10000
    #print('max_timesteps: ', max_timesteps)
    # We start the main loop over max timesteps
    while total_timesteps < max_timesteps:
        # If the episode is done
        if done:

            # If we are not at the very beginning, we start the training process of the model
            if total_timesteps != 0:
                print("Total Timesteps: {} Episode Num: {} Reward: {}".format(
                    total_timesteps, episode_num, episode_reward))
                policy.train(replay_buffer, episode_timesteps, batch_size,
                            discount, tau, policy_noise, noise_clip, policy_freq)

            # We evaluate the episode and we save the policy
            if timesteps_since_eval >= eval_freq:
                timesteps_since_eval %= eval_freq
                evaluations.append(evaluate_policy(policy))
                policy.save(file_name, directory="./pytorch_models")
                np.save("./results/%s" % (file_name), evaluations)

            # When the training step is done, we reset the state of the environment
            obs = env.reset()

            # set done to False
            done = False

            # Set rewards and episode timesteps to zero
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # # find action
        # # for intial timesteps, sample the actions.
        if total_timesteps < start_timesteps:
            action = env.sample_action()
        else:  # After 10000 timesteps, we switch to the model
            action = policy.select_action(np.array(obs))
            action = np.argmax(action) # convert the binary to decimal
            # If the explore_noise parameter is not 0, we add noise to the action and we clip it
            # if expl_noise != 0:
            #     action = (action + \
            #         np.random.normal(0, expl_noise, \
            #             size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)

        # # get the next state and reward
        new_obs, reward, done, info = env.step(action)

        # # We check if the episode is done
        done_bool = 0 if episode_timesteps + \
            1 == max_episode_steps else float(done)

        # # We increase the total reward
        episode_reward += reward

        # # We store the new transition into the Experience Replay memory (ReplayBuffer)
        replay_buffer.add((obs, new_obs, action, reward, done_bool))

        # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
        obs = new_obs
        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1


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
        global env

        print('Build the Game')
        #parent = game
        parent = Game()
        env = parent
        parent.init_game()
        parent.serve_car()

        print('Init the map')
        #init_map()

        print('Painting the canvas')
        self.painter = MyPaintWidget()

        # train the model
        #Clock.schedule_once(parent.train)
        #Clock.schedule_once(new_train)
         

        ###-- buttons ---##
        # clearbtn = Button(text = 'clear')
        # savebtn = Button(text = 'save', pos = (parent.width, 0))
        
        # clearbtn.bind(on_release = self.clear_canvas)
        # savebtn.bind(on_release = self.save)
        
        # parent.add_widget(self.painter)
        # parent.add_widget(clearbtn)
        # parent.add_widget(savebtn)
        
        # Train Button
        trainBtn = Button(text = 'Train')
        trainBtn.bind(on_release = self.train)
        parent.add_widget(trainBtn)

        # Load Button
        loadbtn = Button(text='load', pos=(parent.width, 0))
        loadbtn.bind(on_release=self.load)
        parent.add_widget(loadbtn)

        return parent

 
    def train(self, obj):
        global evaluations
        #evaluations = [evaluate_policy(policy)]
        print('Evaluating Policy...')
        evaluations = [evaluate_policy(policy, eval_episodes=2)]
        print('Start Training...')
        Clock.schedule_once(new_train)
        print('Training Done!')
        #verify_train()

    def load(self, obj):
        print('Loading models...')
        file_name = 'car_td3'
        policy = TD3(state_dim, action_dim, max_action)
        policy.load(file_name, './pytorch_models/')
        _ = evaluate_policy(policy, eval_episodes=2)
        print('Loading Done!')

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))
        
    def save(self, obj):
        print("saving brain...")
        #brain.save()
        plt.plot(scores)
        plt.show()

   # def load(self, obj):
   #     print("loading last saved brain...")
        #brain.load()

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
   

