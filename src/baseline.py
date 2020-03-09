import os
import sys

import gym
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import config


import learningwithoutdata

DATA_PATH = "../data/synthetic_data/first_dataset"


def main():
    env = gym.make(
        "teaching-env-v0",
        teacher_path=os.path.join(
            os.getcwd(),
            "../saved_models",
            sys.argv[1]
        ),
        validation_path=DATA_PATH)
    obs = env.reset()

    total_reward = float('-inf')
    prog = tqdm(
        range(config.MAX_QUERIES),
        postfix={'Reward': total_reward}
    )

    # Plot student's initialized predicted function
    '''inputs = np.linspace(-5, 5, num=1000)
    outputs = env.student_model(inputs.reshape(-1, 1))
    plt.scatter(inputs, outputs, s=0.1, label='Initialized')
    plt.title("Initialized Student's Approximation")
    plt.savefig('./visualizations/functions/initialized')
    plt.clf()'''

    actions = [] # For visualization
    total_reward = 0.0
    for i in prog:
        action = np.random.uniform(-5, 5, size=(1,))
        obs, reward, done, info = env.step(action)
        total_reward += reward
        prog.set_postfix({'Reward': total_reward})
        actions.append(np.asscalar(action)) 
    plt.hist(actions, bins=config.NUM_BINS, range=(-5, 5), density=True)
    plt.savefig('./visualizations/histograms/random')
    plt.clf()

    # Plot teacher's function
    '''plt.scatter(env.validation_inputs.reshape(-1), env.validation_targets.reshape(-1), s=0.1, label='Teacher')
    plt.title("Teacher Function")
    plt.savefig('./visualizations/functions/teacher')
    plt.clf()'''

    # Plot student's predicted function
    inputs = np.linspace(-5, 5, num=1000)
    outputs = env.student_model(inputs.reshape(-1, 1))
    plt.scatter(inputs, outputs, s=0.1, label='Random')
    plt.title("Random Student's Approximation")
    plt.ylim((-60, 100))
    plt.savefig('./visualizations/functions/random')
    plt.clf()






if __name__ == "__main__":
    main()
