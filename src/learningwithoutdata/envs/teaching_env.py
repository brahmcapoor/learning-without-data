import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import os

import tensorflow as tf

from models.student_model import StudentModel
from models.teacher_model import TeacherModel


class TeachingEnv(gym.Env):

    def __init__(self, teacher_path, validation_path, max_queries=100, student_lr=0.2):
        self.sess = tf.Session()
        self.teacher_path = teacher_path
        self.student_lr = student_lr

        self.validation_inputs = np.load(os.path.join(
            validation_path, "inputs.npz")
        ).reshape(-1, 1)
        self.validation_targets = np.load(os.path.join(
            validation_path, "targets.npz")
        ).reshape(-1, 1)

        self.student_queries = 0
        self.max_queries = max_queries
        self.teacher_model = TeacherModel(
            input_dim=1,
            target_dim=1,
            layers=[12, 12, 8],
            activation=tf.nn.sigmoid,
            lr=1e-4
        )
        self.teacher_model.load(self.teacher_path)

        self.student_model = StudentModel(
            input_dim=1,
            target_dim=1,
            layers=[4, 4],
            activation=tf.nn.sigmoid,
            lr=self.student_lr # TODO WOODY CHANGE BACK TO 0.01 if this doesn't work
        )
        # self.sess.run(tf.global_variables_initializer())

        self.num_queries = 1  # TODO
        self.action_space_low = -5 # TODO
        self.action_space_high = 5  # TODO
        self.observation_space_low = -np.inf  # TODO
        self.observation_space_high = np.inf  # TODO

        self.action_space = spaces.Box(
            low=self.action_space_low,
            high=self.action_space_high,
            shape=(self.num_queries, )  # number of examples we're querying
        )

        # WOODY ATTEMPT 1 STATE
        '''self.observation_space = spaces.Box(
            low=self.observation_space_low,
            high=self.observation_space_high,
            shape=(self.student_model.num_weights + 4, )
        )'''
        self.history_len = 10
        self.observation_space = spaces.Box(
            low=self.observation_space_low,
            high=self.observation_space_high,
            shape=(4 * self.history_len, )
        )

        _, self.previous_val_loss = self.student_model.validate(
            self.validation_inputs,
            self.validation_targets
        )

        self.state_queue = [0.0, 0.0, 0.0, 0.0] * self.history_len

    def step(self, action):
        """
        Action is a (1,) numpy array
        """
        action = action.reshape(-1, 1)  # dimensionality case
        teacher_output = self.teacher_model(action)
        student_output, MSE_loss, _ = self.student_model.train_step(
            action, teacher_output
        )
        _, validation_loss = self.student_model.validate(
            self.validation_inputs,
            self.validation_targets
        )
        '''obs = np.hstack(
            [
                self.student_model.weights,
                np.array(
                    [
                        np.asscalar(action),
                        np.asscalar(teacher_output),
                        np.asscalar(student_output),
                        MSE_loss
                    ]
                )
            ]
        )'''
        # WOODY ATTEMPT 1 STATE
        self.state_queue.append(np.asscalar(action))
        self.state_queue.append(np.asscalar(teacher_output))
        self.state_queue.append(np.asscalar(student_output))
        self.state_queue.append(MSE_loss)
        for i in range(4):
            self.state_queue.pop(0)
        obs = np.asarray(self.state_queue)

        info = {}
        self.student_queries += 1
        done = (self.student_queries == self.max_queries)
        # WOODY ATTEMPT 1
        reward = self.previous_val_loss - validation_loss
        self.previous_val_loss = validation_loss

        # state, reward, isterminal, metadata
        return (obs, reward, done, info)

    def reset(self):
        """
        Upon reaching end of episode, go back to
        initialization state
        """
        self.student_queries = 0
        # WOODY ATTEMPT 1 STATE
        #obs = np.hstack([self.student_model.weights, np.zeros((4,))])
        self.state_queue = [0.0, 0.0, 0.0, 0.0] * self.history_len
        obs = np.zeros((4 * self.history_len,))
        tf.reset_default_graph()
        self.teacher_model = TeacherModel(
            input_dim=1,
            target_dim=1,
            layers=[12, 12, 8],
            activation=tf.nn.sigmoid,
            lr=1e-4
        )
        self.teacher_model.load(self.teacher_path)

        self.student_model = StudentModel(
            input_dim=1,
            target_dim=1,
            layers=[4, 4],
            activation=tf.nn.sigmoid,
            lr=self.student_lr
        )
        _, self.previous_val_loss = self.student_model.validate(
            self.validation_inputs,
            self.validation_targets
        )
        return obs

    def render(self, mode='human', close=False):
        pass
