import gym
import numpy as np

import learningwithoutdata


def main():
    env = gym.make("teaching-env-v0")

    obs = env.reset()
    action = np.array([42])
    obs, rewards, dones, info = env.step(action)
    env.render()


if __name__ == "__main__":
    main()
