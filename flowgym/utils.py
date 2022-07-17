"""Utilities for the captain gym environment"""

import numpy as np
import matplotlib.colors
import matplotlib.cm


def generate_perlin_noise_2d(shape, res=(2, 2), low=0, high=1):
    """generate some perlin noise"""
    # based on https://pvigier.github.io/2018/06/13/perlin-noise-numpy.html
    def f(t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    result = np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)
    result -= result.min()
    result /= result.max()
    result *= high - low
    result += low
    return result


def generate_uv(shape=(256, 256), low=-1, high=1):
    """generate a uv map with 2 random patterns"""
    u = generate_perlin_noise_2d(shape=shape[:2], low=low, high=high)
    v = generate_perlin_noise_2d(shape=shape[:2], low=low, high=high)
    uv = np.dstack([u, v])
    return uv.astype(np.float32)


def velocity2rgb(velocity):
    """convert velocity to rgb"""
    direction = np.arctan2(velocity[..., 1], velocity[..., 0])
    speed = np.sqrt((velocity**2).sum(axis=-1))

    N_direction = matplotlib.colors.Normalize(-np.pi, np.pi)
    N_speed = matplotlib.colors.Normalize(0, np.sqrt(2))

    cmap_direction = matplotlib.cm.hsv
    cmap_speed = matplotlib.cm.gray

    rgba_speed = cmap_speed(N_speed(speed))
    hsv_speed = matplotlib.colors.rgb_to_hsv(rgba_speed[:, :, :3])

    rgba_direction = cmap_direction(N_direction(direction))
    hsv_direction = matplotlib.colors.rgb_to_hsv(rgba_direction[:, :, :3])
    hsv = np.dstack([hsv_direction[..., 0], hsv_direction[..., 1], hsv_speed[..., 2]])
    rgb = matplotlib.colors.hsv_to_rgb(hsv)
    return rgb
