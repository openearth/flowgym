import gym
from stable_baselines3 import PPO
import gym.utils.env_checker
import gym.utils.env_checker
import flowgym


def train_world_env_with_stable_baselines():
    env = gym.make('flowgym/WorldEnv-v0', obs_layout_dict=True)
    env.reset()

    tensor_board_log_dir = "logs/WorldEnv-v0"
    policy = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=tensor_board_log_dir)

    policy_dir = "policy_dir/WorldEnv-v0"
    time_steps = 10000
    for i in range(100):
        policy.learn(total_timesteps=time_steps, reset_num_timesteps=False, tb_log_name="PPO")
        current_policy_dir = f"{policy_dir}/{time_steps * (i + 1)}"
        policy.save(current_policy_dir)


def test_world_env_with_stable_baselines():
    # Load the last model
    policy_dir = "policy_dir/WorldEnv-v0"
    current_policy_dir = f"{policy_dir}/{850000}"

    # load policy
    env = gym.make('flowgym/WorldEnv-v0')
    env.reset()
    policy = PPO.load(current_policy_dir, env=env)

    # Execute a number of episodes
    episodes = 200
    for ep in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            # pass observation to model to get predicted action
            action, _states = policy.predict(obs)
            # pass action to env and get info back
            obs, rewards, done, info = env.step(action)
            env.render(mode='human')
