"""Top-level package for flowgym."""

__author__ = """Fedor Baart"""
__email__ = "fedor.baart@deltares.nl"
__version__ = "0.1.0"


from gym.envs.registration import register

register(
    id=f"flowgym/FlowWorldEnv-v0",
    entry_point="flowgym.envs.FlowWorldEnv:FlowWorldEnv",
)

register(
    id=f"flowgym/WorldEnv-v0",
    entry_point="flowgym.envs.WorldEnv:WorldEnv",
)
