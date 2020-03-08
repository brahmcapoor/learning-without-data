import os
import sys

import gym
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
# import matplotlib.pyplot as plt

import learningwithoutdata
import config

DATA_PATH = "../data/synthetic_data/first_dataset"

def main():
    env = gym.make(
        "teaching-env-v0",
        teacher_path=os.path.join(
            os.getcwd(),
            "../saved_models",
            sys.argv[1]
        ),
        validation_path=DATA_PATH,
        max_queries=config.MAX_QUERIES)
    agent_model = PPO2(MlpPolicy, env, verbose=1)
    agent_model.learn(total_timesteps=config.MAX_QUERIES * config.NUM_TRAIN_EPISODES)

    obs = env.reset()

    reward = float('-inf')
    prog = tqdm(
        range(config.MAX_QUERIES),
        postfix={'Reward': reward}
    )

    actions = [] # For visualization
    for i in prog:
        action, _states = agent_model.predict(obs)
        obs, reward, done, info = env.step(action)
        prog.set_postfix({'Reward': reward})
        actions.append(np.asscalar(action)) 
    # plt.hist(actions, bins=config.NUM_BINS, range=(-5, 5), density=True)
    # plt.savefig('./visualizations/histograms/PPO2')
    # plt.clf()

    # Plot student's predicted function
    inputs = np.linspace(-5, 5, num=1000)
    outputs = env.student_model(inputs.reshape(-1, 1))
    # plt.scatter(inputs, outputs, s=0.1, label='PPO2')
    # plt.title("PPO2 Student's Approximation")
    # plt.savefig('./visualizations/functions/PPO2')
    # plt.clf()

if __name__ == "__main__":
    main()
