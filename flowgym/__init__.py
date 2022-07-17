"""Top-level package for flowgym."""

__author__ = """Fedor Baart"""
__email__ = "fedor.baart@deltares.nl"
__version__ = "0.1.0"


from gym.envs.registration import register

register(
    id=f"flowgym/FlowWorldEnv-v{__version__}",
    entry_point="flowgym.envs.flow2d:FlowWorldEnv",
)
