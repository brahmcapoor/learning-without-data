import os
import sys

import gym
import numpy as np
import tensorflow as tf
from tqdm import tqdm

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

    reward = float('-inf')
    prog = tqdm(
        range(2000),
        postfix={'Reward': reward}
    )

    for i in prog:
        action = np.random.uniform(-5, 5, size=(1,))
        obs, reward, done, info = env.step(action)
        prog.set_postfix({'Reward': reward})


if __name__ == "__main__":
    main()
