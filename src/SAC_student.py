import os
import sys

import gym
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC
import matplotlib.pyplot as plt

import learningwithoutdata
import config

DATA_PATH = "../data/synthetic_data/first_dataset"

def select_action(agent_model, obs, epsilon=config.EPSILON_EXPLORATION):
    action = None
    if np.random.rand() <= epsilon:
        action =  np.random.uniform(-5, 5, size=(1,))
    else:
        action, _states = agent_model.predict(obs, deterministic=False)
    return action


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
    agent_model = SAC(MlpPolicy, env, train_freq=1, batch_size=64, learning_rate=3e-4, learning_starts=0, buffer_size=1000, random_exploration=config.EPSILON_EXPLORATION, gamma=config.GAMMA, verbose=1)
    #agent_model.learn(total_timesteps=config.MAX_QUERIES * config.NUM_TRAIN_EPISODES)
    #agent_model.save('test_SAC')

    agent_model.load('test_SAC', env=env)

    obs = env.reset()

    total_reward = float('-inf')
    prog = tqdm(
        range(config.MAX_QUERIES),
        postfix={'Reward': total_reward}
    )

    actions = [] # For visualization
    total_reward = 0.0
    for i in prog:
        action = select_action(agent_model, obs, epsilon=config.EPSILON_EXPLORATION)
        #action, _states = agent_model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        prog.set_postfix({'Reward': total_reward})
        actions.append(np.asscalar(action)) 
    plt.hist(actions, bins=config.NUM_BINS, range=(-5, 5), density=True)
    plt.savefig('./visualizations/histograms/SAC')
    plt.clf()

    # Plot student's predicted function
    inputs = np.linspace(-5, 5, num=1000)
    outputs = env.student_model(inputs.reshape(-1, 1))
    plt.scatter(inputs, outputs, s=0.1, label='SAC')
    plt.title("SAC Student's Approximation")
    plt.ylim((-60, 100))
    plt.savefig('./visualizations/functions/SAC')
    plt.clf()

if __name__ == "__main__":
    main()
