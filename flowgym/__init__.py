"""Top-level package for flowgym."""

__author__ = """Fedor Baart"""
__email__ = "fedor.baart@deltares.nl"
__version__ = "0.1.0"


from gym.envs.registration import register

register(
    id="flowgym/FlowWorldEnv",
    entry_point="flowgym.envs.flow2d:FlowWorldEnv",
)
