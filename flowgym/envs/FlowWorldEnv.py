"""A gym environment with velocity field"""

import gym
import numpy as np
import matplotlib.pyplot as plt
from flowgym.utils import generate_uv


class FlowWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, grid_size=256, velocity=None, normalize=True):
        # resolution is always 1m
        # The size of the grid in m
        super().__init__()
        self.grid_size = grid_size
        self.normalize = normalize

        # store rendering thing in here
        self.mpl = {}

        # slice for quivers
        self._quiver_s = np.s_[::8, ::8]

        self.dtype = np.float32

        # maximum velocity
        self.velocity_min = -1
        self.velocity_max = 1

        # default to 0 velocity field
        if velocity is None:
            velocity = generate_uv(
                (self.grid_size, self.grid_size), np_random=self.np_random
            )

        # hidden property for velocity, as the normal way to get access is to
        self._velocity = velocity

        # Observations are dictionaries with the agent's and the target's location.
        observation_dict = {
            # # object position
            "agent": gym.spaces.Box(
                low=0, high=grid_size, shape=(2,), dtype=self.dtype
            ),
            # # target position
            "target": gym.spaces.Box(
                low=0, high=grid_size, shape=(2,), dtype=self.dtype
            ),
            "velocity": gym.spaces.Box(
                low=self.velocity_min,
                high=self.velocity_max,
                shape=(grid_size, grid_size, 2),
                dtype=self.dtype,
            ),
        }

        self.observation_space = gym.spaces.Dict(observation_dict)

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = gym.spaces.Discrete(8)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        sqrt2 = np.sqrt(2)
        self._action_to_direction = {
            0: np.array([1, 0], dtype=self.dtype),
            1: np.array([0, 1], dtype=self.dtype),
            2: np.array([-1, 0], dtype=self.dtype),
            3: np.array([0, -1], dtype=self.dtype),
            4: np.array([sqrt2, sqrt2], dtype=self.dtype),
            5: np.array([sqrt2, -sqrt2], dtype=self.dtype),
            6: np.array([-sqrt2, sqrt2], dtype=self.dtype),
            7: np.array([-sqrt2, -sqrt2], dtype=self.dtype),
        }

    def _get_obs(self):
        obs = {
            "agent": self._agent_position,
            "target": self._target_position,
            "velocity": self._velocity,
        }
        if self.normalize:
            obs["agent"] = self._agent_position / self.grid_size
            obs["target"] = self._target_position / self.grid_size
            obs["velocity"] = self._velocity / self.grid_size
        return obs

    def _get_info(self):
        distance = np.sqrt(np.sum((self._target_position - self._agent_position) ** 2))
        if self.normalize:
            distance /= self.grid_size
        return {"distance": distance}

    def reset(self, seed=None, return_info=False, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random

        # agent_shape = self.observation_space["agent"].shape
        self._agent_position = self.np_random.uniform(low=0, high=self.grid_size, size=(2,)).astype(self.dtype)

        # We will sample the target's location randomly until it does not coincide with the agent's location

        self._target_position = self._agent_position
        # make sure we don't put target on agent position. If position is different we're done
        while np.array_equal(self._target_position, self._agent_position):
            self._target_position = self.np_random.uniform(
                low=0, high=self.grid_size, size=(2,)
            ).astype(self.dtype)

        observation = self._get_obs()
        info = self._get_info()
        # reset render
        self.close()

        return (observation, info) if return_info else observation

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]

        info = self._get_info()
        distance_0 = info["distance"]

        # We use `np.clip` to make sure we don't leave the grid
        self._agent_position = np.clip(
            self._agent_position + direction, 0, self.grid_size
        ).astype(self.dtype)

        info = self._get_info()
        distance_1 = info["distance"]
        # An episode is done if the agent has reached the target
        done = bool(distance_1 < 1)
        # Binary sparse rewards
        reward = distance_0 - distance_1

        observation = self._get_obs()

        return observation, reward, done, info

    def create_render(self):
        """create a matplotlib figure, with elements stored in a dictionary"""
        mpl = self.mpl
        fig, ax = plt.subplots()
        obs = self._get_obs()
        mpl["fig"], mpl["ax"] = fig, ax

        (mpl["target"],) = ax.plot(*obs["target"], "rx", alpha=1)
        (mpl["agent"],) = ax.plot(*obs["agent"], "ro", alpha=1)

        XY = np.mgrid[: self.grid_size, : self.grid_size]
        if self.normalize:
            XY = XY / self.grid_size
        X, Y = XY

        s = self._quiver_s
        U = obs["velocity"][..., 0]
        V = obs["velocity"][..., 1]
        # scale is inverted
        mpl["quiver"] = ax.quiver(X[s], Y[s], U[s], V[s], units="xy", scale=0.1)
        return mpl

    def update_render(self):
        """update pre-rendered figure"""
        mpl = self.mpl
        obs = self._get_obs()
        s = self._quiver_s
        mpl["quiver"].set_UVC(obs["velocity"][s][..., 0], obs["velocity"][s][..., 1])
        mpl["target"].set_data(*obs["target"])
        mpl["agent"].set_data(*obs["agent"])

    def render(self, mode="human"):
        """render the current timestep. In human mode on first call don't return the figure, as it will be already shown using the subplots function."""
        assert mode in (
            "human",
            "rgb_array",
        ), f"we only render in human mode, got {mode}"

        # get the rendered figure
        mpl = self.mpl

        created = False
        # if we don't have a figure yet, create it
        if not mpl:
            self.create_render()
            created = True

        # render our figure
        self.update_render()

        fig = mpl["fig"]

        if mode == "human":
            # first call already renders the figure
            if not created:
                return fig
            else:
                return

        elif mode == "rgb_array":
            buf = fig.canvas.buffer_rgba()
            img = np.asarray(buf)
            return img

    def close(self):
        """close the figure"""
        fig = self.mpl.get("fig")
        if fig:
            plt.close(fig)
        self.mpl = {}
