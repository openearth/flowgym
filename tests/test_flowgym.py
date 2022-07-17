#!/usr/bin/env python

"""Tests for `flowgym` package."""

import pytest

from click.testing import CliRunner

import flowgym

import gym.utils.env_checker
import gym


@pytest.fixture
def env():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    env = gym.make("flowgym/FlowWorldEnv")
    return env


def test_env(env):
    """Run the env checker"""
    gym.utils.env_checker.check_env(env)
