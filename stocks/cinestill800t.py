import numpy as np
from .base import apply_color_matrix, hd_curve, apply_halation, apply_grain


def process(linear):
    # Tungsten-balanced shot in daylight: strong blue/teal cast
    # Push blue, pull red, shift greens toward teal
    color_matrix = np.array(
        [
            [ 0.82,  0.05,  0.13],
            [-0.02,  0.92,  0.10],
            [ 0.00,  0.08,  1.15],
        ],
        dtype=np.float32,
    )
    img = apply_color_matrix(linear, color_matrix)

    # C41 negative: per-channel H&D with teal lift in shadows
    r = hd_curve(img[:, :, 0], gain=0.88, toe_pow=0.92, shoulder=0.70, lift=0.025)
    g = hd_curve(img[:, :, 1], gain=0.90, toe_pow=0.93, shoulder=0.71, lift=0.030)
    b = hd_curve(img[:, :, 2], gain=0.92, toe_pow=0.90, shoulder=0.68, lift=0.040)
    img = np.stack([r, g, b], axis=2)
    # Defining Cinestill feature: massive red halation from remjet removal
    # Very strong, wide, almost entirely red channel
    img = apply_halation(img, radius=35, strength=0.18, warm=True)

    # ISO 800, chunky color grain
    img = apply_grain(img, fine_sigma=1.6, coarse_sigma=4.0, strength=0.038)

    return img
