from gym.envs.registration import register

register(
    id='teaching-env-v0',
    entry_point='learningwithoutdata.envs:TeachingEnv'
)

register(
    id='teaching-env-discrete-v0',
    entry_point='learningwithoutdata.envs:TeachingEnvDiscrete'
)
