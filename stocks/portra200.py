import numpy as np
from .base import apply_color_matrix, hd_curve, apply_halation, apply_grain


def process(linear):
    color_matrix = np.array(
        [
            [ 1.06, -0.03, -0.03],
            [-0.02,  0.97,  0.05],
            [-0.04,  0.01,  1.03],
        ],
        dtype=np.float32,
    )
    img = apply_color_matrix(linear, color_matrix)
    r = hd_curve(img[:, :, 0], gain=0.93, toe_pow=0.88, shoulder=0.72, lift=0.025)
    g = hd_curve(img[:, :, 1], gain=0.88, toe_pow=0.90, shoulder=0.70, lift=0.020)
    b = hd_curve(img[:, :, 2], gain=0.82, toe_pow=0.92, shoulder=0.68, lift=0.030)
    img = np.stack([r, g, b], axis=2)
    img = apply_halation(img, radius=16, strength=0.045, warm=True)
    img = apply_grain(img, fine_sigma=1.1, coarse_sigma=0, strength=0.022)
    img[:, :, 0] = np.clip(img[:, :, 0] * 1.012, 0, 1)
    img[:, :, 2] = np.clip(img[:, :, 2] * 0.980, 0, 1)
    return img
