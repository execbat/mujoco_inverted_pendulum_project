from gymnasium.envs.registration import register

register(
    id="SberInvertedPendulum-v0",
    entry_point="inv_pendulum_gym_env_0:InvertedPendulumGymEnv_0",  
    max_episode_steps=2000
)

