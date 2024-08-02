from gym.envs.registration import register

register(
    id='VideoStreaming-v0',
    entry_point='media_env.envs:VideoStreamingEnv',
)
