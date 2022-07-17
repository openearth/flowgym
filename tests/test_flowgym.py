#!/usr/bin/env python

"""Tests for `flowgym` package."""

import pytest
import numpy as np

from click.testing import CliRunner
import gym.utils.env_checker
import gym

import flowgym


@pytest.fixture
def env():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    env = gym.make("flowgym/FlowWorldEnv")
    return env


def test_check_env(env):
    """Run the env checker"""
    gym.utils.env_checker.check_env(env)


def test_has_velocities(env):
    """Run the env checker"""
    assert isinstance(
        env.unwrapped._velocity, np.ndarray
    ), f"velocities should be an array, got {type(env.unwrapped._velocities)}"
