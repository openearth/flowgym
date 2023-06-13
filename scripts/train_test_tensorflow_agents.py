import platform
import tensorflow as tf

if platform.uname().system == 'Windows':
    tf.config.set_visible_devices([], 'GPU')

import gym
import gym.utils.env_checker
import tensorflow as tf
import tf_agents.environments.gym_wrapper
import tf_agents.networks
import tf_agents.agents.dqn
import tf_agents.replay_buffers
import tf_agents.drivers
import flowgym

def test_world_env():
    env = gym.make('flowgym/WorldEnv-v0')
    env.reset()

    # wrap OpenAI Gym
    py_env = tf_agents.environments.gym_wrapper.GymWrapper(gym_env=env)
    tf_env = tf_agents.environments.tf_py_environment.TFPyEnvironment(py_env)

    # create neural network
    q_net = tf_agents.networks.q_network.QNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        fc_layer_params=(100,),
        dtype=env.dtype)
    learning_rate = 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    td_errors_loss = tf_agents.utils.common.element_wise_squared_loss

    # make agent
    agent = tf_agents.agents.dqn.dqn_agent.DqnAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=td_errors_loss,
        train_step_counter=tf.Variable(0))
    agent.initialize()

    # replay buffer
    replay_buffer_capacity = 1000
    replay_buffer = tf_agents.replay_buffers.tf_uniform_replay_buffer.TFUniformReplayBuffer(
        agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=replay_buffer_capacity
    )

    # dataset
    dataset = replay_buffer.as_dataset(sample_batch_size=tf_env.batch_size, num_steps=2)
    iterator = iter(dataset)

    # train
    num_train_steps = 100
    losses = []
    collect_steps_per_iteration = 1000
    for _ in range(num_train_steps):
        tf_env.reset()
        for i in range(collect_steps_per_iteration):
            time_step = tf_env.current_time_step()
            action_step = agent.policy.action(time_step)
            next_time_step = tf_env.step(action_step.action)
            traj = tf_agents.trajectories.trajectory.from_transition(time_step, action_step, next_time_step)

            # Add trajectory to the replay buffer
            replay_buffer.add_batch(traj)

        experience, _ = next(iterator)
        loss = agent.train(experience=experience)
        losses.append(loss)

    # test
    actions = []
    env.reset()
    step = tf_env.reset()
    steps = [step]
    for i in range(100):
        action = agent.policy.action(step)
        step = tf_env.step(action)
        actions.append(action)
        steps.append(step)
        env.render(mode='human')
