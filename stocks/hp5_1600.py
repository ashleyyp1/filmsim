import numpy as np
from .base import apply_grain


def process(linear):
    # Panchromatic B&W conversion — NOT just desaturate
    luma = 0.25 * linear[:, :, 0] + 0.68 * linear[:, :, 1] + 0.07 * linear[:, :, 2]

    # Pushed H&D: hard shadow blocking, snappy midtones
    x = np.clip(luma, 0, 1)
    shadow_threshold = 0.12
    toe_region = x < shadow_threshold
    toe_val = (x / shadow_threshold) ** 1.6 * shadow_threshold * 0.6
    t = np.clip((x - shadow_threshold) / (1.0 - shadow_threshold), 0, 1)
    mid_val = t * t * (3.0 - 2.0 * t)
    blended = (t * 0.4 + mid_val * 0.6) * 1.35
    s = np.clip((blended - 0.80) / 0.20, 0, 1)
    blended = blended * (1.0 - s * s * 0.25)
    density = np.where(
        toe_region,
        toe_val,
        shadow_threshold * 0.6 + blended * (1.0 - shadow_threshold),
    )
    density = np.clip(density, 0, 1)

    density = apply_grain(density, fine_sigma=1.8, coarse_sigma=4.5, strength=0.048)

    # B&W halation (neutral)
    highlights = np.clip(density - 0.82, 0, 1) ** 2.0
    from scipy.ndimage import gaussian_filter
    halo = gaussian_filter(highlights, sigma=10)
    density = np.clip(density + halo * 0.018, 0, 1)

    # Split tone: cool shadows, warm highlights
    r = density * (0.97 + density * 0.03)
    g = density * 0.99
    b = density * (1.02 - density * 0.04)
    return np.stack([np.clip(r, 0, 1), np.clip(g, 0, 1), np.clip(b, 0, 1)], axis=2)
