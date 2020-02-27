import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np

import tensorflow as tf

from models.student_model import StudentModel
from models.teacher_model import TeacherModel


class TeachingEnv(gym.Env):

    def __init__(self):

        self.teacher_model = TeacherModel(
            input_dim=1,
            target_dim=1,
            layers=[12, 12, 8],
            activation=tf.nn.sigmoid,
            lr=1e-4
        )
        self.teacher_model.load(
            '/Users/brahm/Dev/classes/cs234/learning-without-data/src/'
        )

        self.student_model = StudentModel(
            input_dim=1,
            target_dim=1,
            layers=[4, 4],
            activation=tf.nn.sigmoid,
            lr=0.01
        )

        self.num_queries = 1  # TODO
        self.action_space_low = -np.inf  # TODO
        self.action_space_high = np.inf  # TODO
        self.observation_space_low = -np.inf  # TODO
        self.observation_space_high = np.inf  # TODO

        self.action_space = spaces.Box(
            low=self.action_space_low,
            high=self.action_space_high,
            shape=(self.num_queries, )  # number of examples we're querying
        )

        self.observation_space = spaces.Box(
            low=self.observation_space_low,
            high=self.observation_space_high,
            shape=(self.student_model.get_num_weights(), )
        )

    def step(self, action):
        obs = None
        reward = None
        done = False
        info = {}
        return (obs, reward, done, info)  # state, reward, isterminal, metadata

    def reset(self):
        """
        Upon reaching end of episode, go back to 
        initialization state
        """
        obs = None
        return None

    def render(self, mode='human', close=False):
        print("TODO: Render stuff here")
