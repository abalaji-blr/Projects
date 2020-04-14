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
from td3_ai import ReplayBuffer, TD3
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
    goal_x = 1420  # target x
    goal_y = 622   # target y
    first_update = False
    swap = 0


# Initializing the last distance
last_distance = 0
#--------------------------- Training Related ---------------------------
total_timesteps = 0
timesteps_since_eval = 0
episode_num = 0
done = True
t0 = time.time()

replay_buffer = ReplayBuffer(100)

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

    def reset(self):
        global longueur
        global largeur

        #self.car.center = self.center
        #self.car.center  = (600,310)
        self.car.x = 600
        self.car.y = 350
        longueur = self.width
        largeur = self.height
        obs = self.get_sand_image_patch(IMAGE_PATCH_WIDTH, IMAGE_PATCH_HEIGHT)
        return(obs)

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

    #def step(self, dt): # dt - kivy - delta-time coming from clock
    def step(self, action):
        global goal_x
        global goal_y

        print('car location: ', self.car.x, self.car.y)
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx, yy))/180.
        sand_patch = self.get_sand_image_patch(
            IMAGE_PATCH_WIDTH, IMAGE_PATCH_HEIGHT)
        print('image_patch: ', sand_patch.shape)

        img_patch = self.get_image_patch(IMAGE_PATCH_WIDTH, IMAGE_PATCH_HEIGHT)
        patch = PILImage.fromarray(img_patch)
        #patch.save('./patch.png')
        new_obs = sand_patch
        #reward

    def train(self, dt):
        global done
        global total_timesteps

        print('replay_buffer size:', replay_buffer.max_size)
        for i in range(5):
            self.step(1)
        max_timesteps = 5

        # We start the main loop over max timesteps
        while total_timesteps < max_timesteps:

            # If the episode is done
            if done:
                obs = self.reset()

            # set done to False
            done = False

            # get the next state and reward
            #new_obs, reward, done, info = self.step(1)
            total_timesteps += 1


                

    def serve_car(self):
        self.car.center = self.center
        self.car.x = 600
        self.car.y = 350
        print('car location:', self.car.center)
        self.car.velocity = Vector(6, 0)
        self.reward = 0.0
        self.prev_reward = 0.0
        ## actions - steer, gas, brake
        ## steer: range -1 to 1 => -1 left, 0 straight, > 0 right
        self.action_space = spaces.Box(
            np.array([-1, 0, 0]), np.array([+1, +1, +1]), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(IMAGE_PATCH_HEIGHT, IMAGE_PATCH_WIDTH), dtype=np.uint8)

    def update(self, dt):

        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
        global swap

        longueur = self.width
        largeur = self.height

        if first_update:
            init_map()

        print('car location: ', self.car.x, self.car.y)
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx, yy))/180.
        last_signal = [self.car.signal1, self.car.signal2,
                       self.car.signal3, orientation, -orientation]
        #action = brain.update(last_reward, last_signal)

        #scores.append(brain.score())

        # rotation = action2rotation[action]
        # self.car.move(rotation)

        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        # self.ball1.pos = self.car.sensor1
        # self.ball2.pos = self.car.sensor2
        # self.ball3.pos = self.car.sensor3

        if sand[int(self.car.x), int(self.car.y)] > 0:
            self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
            print(1, goal_x, goal_y, distance, int(self.car.x), int(
                self.car.y), im.read_pixel(int(self.car.x), int(self.car.y)))

            last_reward = -1
        else:  # otherwise
            self.car.velocity = Vector(2, 0).rotate(self.car.angle)
            last_reward = -0.2
            print(0, goal_x, goal_y, distance, int(self.car.x), int(
                self.car.y), im.read_pixel(int(self.car.x), int(self.car.y)))
            if distance < last_distance:
                last_reward = 0.1
            # else:
            #     last_reward = last_reward +(-0.2)

        if self.car.x < 5:
            self.car.x = 5
            last_reward = -1
        if self.car.x > self.width - 5:
            self.car.x = self.width - 5
            last_reward = -1
        if self.car.y < 5:
            self.car.y = 5
            last_reward = -1
        if self.car.y > self.height - 5:
            self.car.y = self.height - 5
            last_reward = -1

        if distance < 25:
            if swap == 1:
                goal_x = 1420
                goal_y = 622
                swap = 0
            else:
                goal_x = 9
                goal_y = 85
                swap = 1
        last_distance = distance






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

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation

        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.
        
        if self.sensor1_x>longueur-10 or self.sensor1_x<10 or self.sensor1_y>largeur-10 or self.sensor1_y<10:
            self.signal1 = 10.
        if self.sensor2_x>longueur-10 or self.sensor2_x<10 or self.sensor2_y>largeur-10 or self.sensor2_y<10:
            self.signal2 = 10.
        if self.sensor3_x>longueur-10 or self.sensor3_x<10 or self.sensor3_y>largeur-10 or self.sensor3_y<10:
            self.signal3 = 10.
        

 

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
