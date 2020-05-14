"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class SuperMarioEnv():
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson

    Observation: 
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24 deg        24 deg
        3	Pole Velocity At Tip      -Inf            Inf
        
    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right
        
        Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees
        Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 60
    }

    def __init__(self, maxFrames, levelMaxX):
        self.action_space = spaces.Discrete(4)

        # observation space: [x,x_vel,y,y_vel,frame,closestEnemyDistance]
        # First level max_X = 1888
        lowerBound = np.array([0,
                         -40,
                         0,
                         -40,
                         0,
                         -levelMaxX], dtype=np.float32)
        upperBound = np.array([2000,
                         40,
                         1000,
                         40,
                         maxFrames,
                         levelMaxX], dtype=np.float32)
        self.observation_space = spaces.Box(lowerBound, upperBound, dtype=np.float32)
        self.state = None

    def reset(self):
        self.state = np.array([])
        self.steps_beyond_done = None
        return np.array(self.state)