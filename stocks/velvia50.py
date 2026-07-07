import numpy as np
from .base import apply_color_matrix, hd_curve, apply_halation, apply_grain


def process(linear):
    # Reversal (slide) film: electric saturation, reds deepen, greens shift yellow-green, blues intensify
    color_matrix = np.array(
        [
            [ 1.20, -0.12, -0.08],
            [ 0.04,  1.28, -0.32],
            [-0.06, -0.06,  1.30],
        ],
        dtype=np.float32,
    )
    img = apply_color_matrix(linear, color_matrix)

    # Reversal process: pure blacks (lift=0), high contrast, abrupt toe, hard highlight clip
    # toe_pow > 1 = steep shadow blocking; shoulder close to 1 = late, hard clipping
    r = hd_curve(img[:, :, 0], gain=1.25, toe_pow=1.35, shoulder=0.88, lift=0.0)
    g = hd_curve(img[:, :, 1], gain=1.25, toe_pow=1.35, shoulder=0.88, lift=0.0)
    b = hd_curve(img[:, :, 2], gain=1.22, toe_pow=1.30, shoulder=0.86, lift=0.0)
    img = np.stack([r, g, b], axis=2)
    # ISO 50: barely-there grain
    img = apply_grain(img, fine_sigma=0.6, coarse_sigma=0, strength=0.008)

    # Minimal halation — slide film has virtually none
    img = apply_halation(img, radius=4, strength=0.005, warm=True)

    return img
