"""A gym environment with velocity field"""

import gym
import numpy as np
import matplotlib.pyplot as plt


class WorldEnv(gym.Env):
    """Simple environment with only agent and target"""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, grid_size=256, obs_layout_dict=False):
        # resolution is always 1m
        # The size of the grid in m
        super(WorldEnv, self).__init__()
        self.grid_size = grid_size
        self.obs_layout_dict = obs_layout_dict

        # store rendering thing in here
        self.mpl = {}
        self.dtype = np.float32

        # Observations are dictionaries with the agent's and the target's location.
        if self.obs_layout_dict:
            observation_dict = {
                # # object position
                "agent": gym.spaces.Box(
                    low=0, high=grid_size, shape=(2,), dtype=self.dtype
                ),
                # # target position
                "target": gym.spaces.Box(
                    low=0, high=grid_size, shape=(2,), dtype=self.dtype
                ),
            }
            self.observation_space = gym.spaces.Dict(observation_dict)
        else:
            self.observation_space = gym.spaces.Box(low=0, high=grid_size, shape=(4,), dtype=self.dtype)

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = gym.spaces.Discrete(4)

        self.positions = []

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0], dtype=self.dtype),
            1: np.array([0, 1], dtype=self.dtype),
            2: np.array([-1, 0], dtype=self.dtype),
            3: np.array([0, -1], dtype=self.dtype),
        }

    def _get_obs(self):
        if self.obs_layout_dict:
            obs = {"agent": self._agent_position, "target": self._target_position}
        else:
            obs = np.r_[self._agent_position, self._target_position]
        return obs

    def reset(self, seed=None, return_info=False, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # reset render
        self.close()
        self.positions = []

        # agent_shape = self.observation_space["agent"].shape
        self._agent_position = self.np_random.uniform(low=0, high=self.grid_size, size=(2,)).astype(self.dtype)

        # We will sample the target's location randomly until it does not coincide with the agent's location

        self._target_position = self._agent_position
        # make sure we don't put target on agent position. If position is different we're done
        while np.array_equal(self._target_position, self._agent_position):
            self._target_position = self.np_random.uniform(
                low=0, high=self.grid_size, size=(2,)
            ).astype(self.dtype)

        self.positions.append(self._agent_position)

        observation = self._get_obs()
        info = self._get_info()

        return (observation, info) if return_info else observation

    def _get_info(self):
        distance = abs(self._target_position[0] - self._agent_position[0]) +\
                   abs(self._target_position[1] - self._agent_position[1])
        return {"distance": distance}

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]

        info = self._get_info()
        distance_0 = info["distance"]

        # We use `np.clip` to make sure we don't leave the grid
        self._agent_position = np.clip(self._agent_position + direction, 0, self.grid_size).astype(self.dtype)
        self.positions.append(self._agent_position)

        info = self._get_info()
        distance_1 = info["distance"]

        # Binary sparse rewards
        done = False
        reward = distance_0 - distance_1 if distance_0 > distance_1 else -self.grid_size
        if distance_1 < 1:
            done = True

        observation = self._get_obs()

        return observation, reward, done, info

    def create_render(self):
        """create a matplotlib figure, with elements stored in a dictionary"""
        mpl = {}
        fig, ax = plt.subplots()
        mpl["fig"], mpl["ax"] = fig, ax
        ax.set_xlim([0, self.grid_size])
        ax.set_ylim([0, self.grid_size])

        (mpl["target"],) = ax.plot(*self._target_position, "rx", alpha=1)
        (mpl["agent"],) = ax.plot(*self._agent_position, "ro", alpha=1)
        x = [xy[0] for xy in self.positions]
        y = [xy[1] for xy in self.positions]
        (mpl["trajectory"],) = ax.plot(x, y, "k-", alpha=0.5)

        # Show the graph without blocking the rest of the program
        plt.show(block=False)
        return mpl

    def update_render(self):
        """update pre-rendered figure"""
        mpl = self.mpl
        mpl["target"].set_data(*self._target_position)
        mpl["agent"].set_data(*self._agent_position)

        x = [xy[0] for xy in self.positions]
        y = [xy[1] for xy in self.positions]
        mpl["trajectory"].set_data(x, y)

    def render(self, mode="human"):
        """render the current timestep. In human mode on first call don't return the figure, as it will be already shown using the subplots function."""
        assert mode in (
            "human",
            "rgb_array",
        ), f"we only render in human mode, got {mode}"

        # if we don't have a figure yet, create it
        if not self.mpl:
            self.mpl = self.create_render()

        # render our figure
        self.update_render()

        # Necessary to view frames before they are unrendered
        plt.pause(0.001)
        info = self._get_info()

        if mode == "rgb_array":
            fig = self.mpl.get("fig")
            buf = fig.canvas.buffer_rgba()
            img = np.asarray(buf)
            return img

    def close(self):
        """close the figure"""
        fig = self.mpl.get("fig")
        if fig:
            plt.close(fig)
        self.mpl = {}
