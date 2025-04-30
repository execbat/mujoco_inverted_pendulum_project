from gymnasium.envs.registration import register

#keeping vertical position + reaching the target in different locations + pole mass is random in range [10;30]
register(
    id="SberInvertedPendulum-v0",
    entry_point="inv_pendulum_gym_env_0:InvertedPendulumGymEnv_0",  
    max_episode_steps=2000
)



