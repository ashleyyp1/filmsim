import numpy as np


def compute_grade(samples):
    """
    Analyze a list of linear-space float32 frames sampled from the video.
    Returns grading parameters for apply_grade() — applied consistently to
    every frame so the grade doesn't flicker.
    """
    # Subsample pixels to keep memory flat on long clips
    pixels = np.concatenate(
        [s.reshape(-1, 3)[::8] for s in samples],
        axis=0,
    ).astype(np.float64)

    # Per-channel percentile levels (robust black/white points)
    black = np.percentile(pixels, 1.5, axis=0).astype(np.float32)
    white = np.percentile(pixels, 98.5, axis=0).astype(np.float32)
    span  = np.maximum(white - black, 0.05)   # guard against nearly-flat clips

    # Gray-world white balance in linear, conservatively clamped.
    # Clamping matters: don't kill an intentionally warm or cool scene.
    means   = pixels.mean(axis=0)
    neutral = float(means.mean())
    wb = np.clip((neutral / (means + 1e-8)).astype(np.float32), 0.80, 1.30)

    # Drive median luminance to ~0.20 (linear middle-gray)
    luma = 0.2126 * pixels[:, 0] + 0.7152 * pixels[:, 1] + 0.0722 * pixels[:, 2]
    med_luma = float(np.percentile(luma, 50))
    exposure = float(np.clip(0.20 / (med_luma + 1e-8), 0.35, 4.0))

    return {'black': black, 'span': span, 'wb': wb, 'exposure': exposure}


def apply_grade(linear, params):
    """Apply pre-computed grade to a single linear float32 frame."""
    out = (linear - params['black']) / params['span']   # auto levels
    out = out * params['wb']                             # white balance
    out = out * params['exposure']                       # exposure
    return np.clip(out, 0, 1).astype(np.float32)
