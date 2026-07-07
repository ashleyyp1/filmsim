import numpy as np
from .base import apply_color_matrix, hd_curve, apply_halation, apply_grain


def process(linear):
    # Most saturated color negative ever: reds go deep, greens go vivid
    # Slightly cooler/more neutral than Portra
    color_matrix = np.array(
        [
            [ 1.14, -0.07, -0.07],
            [-0.10,  1.22, -0.12],
            [-0.04, -0.08,  1.14],
        ],
        dtype=np.float32,
    )
    img = apply_color_matrix(linear, color_matrix)

    # C41 negative: moderate shoulder, slightly lifted blacks, cooler balance
    r = hd_curve(img[:, :, 0], gain=0.95, toe_pow=0.90, shoulder=0.73, lift=0.018)
    g = hd_curve(img[:, :, 1], gain=0.94, toe_pow=0.91, shoulder=0.72, lift=0.018)
    b = hd_curve(img[:, :, 2], gain=0.92, toe_pow=0.91, shoulder=0.71, lift=0.018)
    img = np.stack([r, g, b], axis=2)
    img = apply_halation(img, radius=12, strength=0.025, warm=True)
    img = apply_grain(img, fine_sigma=0.8, coarse_sigma=0, strength=0.014)

    return img
