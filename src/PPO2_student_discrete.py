import os
import sys

import gym
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import learningwithoutdata
import config

DATA_PATH = "../data/synthetic_data/first_dataset"

def main():
    env = gym.make(
        "teaching-env-discrete-v0",
        teacher_path=os.path.join(
            os.getcwd(),
            "../saved_models",
            sys.argv[1]
        ),
        validation_path=DATA_PATH,
        max_queries=config.MAX_QUERIES)
    agent_model = PPO2(MlpPolicy, env, gamma=1.0, n_steps=64, verbose=1)
    # Uncomment if training 
    agent_model.learn(total_timesteps=config.MAX_QUERIES * config.NUM_TRAIN_EPISODES, log_interval=10)
    agent_model.save('test_PPO_discrete')

    # Uncomment if loading model
    #agent_model.load('test_PPO2_discrete')

    obs = env.reset()

    total_reward = float('-inf')
    prog = tqdm(
        range(config.MAX_QUERIES),
        postfix={'Reward': total_reward}
    )

    total_reward = 0.0
    actions = [] # For visualization
    for i in prog:
        action, _states = agent_model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        prog.set_postfix({'Reward': total_reward})
        action = np.asarray([np.asscalar(env.actions[np.asscalar(action)])]) # Shape (1,)
        actions.append(np.asscalar(action)) 
    plt.hist(actions, bins=config.NUM_BINS, range=(-5, 5), density=True)
    plt.savefig('./visualizations/histograms/PPO2_discrete')
    plt.clf()

    # Plot student's predicted function
    inputs = np.linspace(-5, 5, num=1000)
    outputs = env.student_model(inputs.reshape(-1, 1))
    plt.scatter(inputs, outputs, s=0.1, label='PPO2_discrete')
    plt.title("PPO2 Student Discrete's Approximation")
    plt.ylim((-60, 100))
    plt.savefig('./visualizations/functions/PPO2_discrete')
    plt.clf()

if __name__ == "__main__":
    main()
