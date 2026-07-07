import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, ifft2, fftfreq


def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def apply_color_matrix(img, matrix):
    h, w, _ = img.shape
    result = img.reshape(-1, 3) @ matrix.T
    return np.clip(result.reshape(h, w, 3), 0, None)


def hd_curve(x, gain, toe_pow, shoulder, lift):
    """Core H&D curve — works in linear 0-1 space, NOT log space"""
    x = np.clip(x, 0, 1)
    toe = x ** toe_pow
    s = np.clip((x - shoulder) / (1.0 - shoulder), 0, 1)
    shoulder_comp = 1.0 - s * s * (3.0 - 2.0 * s) * 0.28
    out = toe * gain * shoulder_comp
    return np.clip(out * (1.0 - lift) + lift, 0, 1)


def shaped_noise(h, w, sigma, seed):
    """Spatially correlated grain — NOT per-pixel noise"""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((h, w)).astype(np.float32)
    freq_y = fftfreq(h).reshape(-1, 1).astype(np.float32)
    freq_x = fftfreq(w).reshape(1, -1).astype(np.float32)
    freq_mag = np.sqrt(freq_x ** 2 + freq_y ** 2) + 1e-8
    envelope = np.exp(-freq_mag ** 2 / (2 * (sigma / max(h, w)) ** 2))
    out = np.real(ifft2(fft2(noise) * envelope)).astype(np.float32)
    return out / (out.std() + 1e-8)


def apply_halation(img, radius, strength, warm=True):
    highlights = np.clip(img - 0.70, 0, 1) ** 1.5
    if warm:
        halo_r = gaussian_filter(highlights[:, :, 0], sigma=radius)
        halo_g = gaussian_filter(highlights[:, :, 1], sigma=radius * 0.5)
        halo_b = gaussian_filter(highlights[:, :, 2], sigma=radius * 0.2)
        halo = np.stack([halo_r, halo_g, halo_b], axis=2)
    else:
        h_single = gaussian_filter(
            highlights if highlights.ndim == 2 else highlights.mean(axis=2),
            sigma=radius,
        )
        halo = np.stack([h_single, h_single, h_single], axis=2)
    return np.clip(img + halo * strength, 0, 1)


def apply_grain(img, fine_sigma, coarse_sigma, strength, midtone_peak=0.40):
    h, w = img.shape[:2]
    is_color = img.ndim == 3
    luma = (
        (0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2])
        if is_color
        else img
    )
    fine = shaped_noise(h, w, fine_sigma, 42)
    coarse = shaped_noise(h, w, coarse_sigma, 99) if coarse_sigma > 0 else np.zeros((h, w), dtype=np.float32)
    combined = fine * 0.6 + coarse * 0.4
    grain_amp = strength * (
        0.55 * np.exp(-((luma - midtone_peak) ** 2) / (2 * 0.22 ** 2))
        + 0.30 * np.exp(-((luma - 0.18) ** 2) / (2 * 0.18 ** 2))
        + 0.15 * np.exp(-((luma - 0.72) ** 2) / (2 * 0.22 ** 2))
    )
    if is_color:
        # Separate per-layer grain — each dye layer at a different emulsion depth
        grain_r = shaped_noise(h, w, fine_sigma * 0.9, 11) * 0.7 + combined * 0.3
        grain_g = shaped_noise(h, w, fine_sigma * 1.0, 22) * 0.5 + combined * 0.5
        grain_b = shaped_noise(h, w, fine_sigma * 1.2, 33) * 0.6 + combined * 0.4

        # 1-pixel spatial offset between layers — dye clouds don't perfectly align
        grain_r = np.roll(np.roll(grain_r,  1, axis=0),  1, axis=1)
        grain_b = np.roll(np.roll(grain_b, -1, axis=0), -1, axis=1)

        result = np.stack(
            [
                img[:, :, 0] + grain_r * grain_amp * 0.85,
                img[:, :, 1] + grain_g * grain_amp * 0.90,
                img[:, :, 2] + grain_b * grain_amp * 1.40,
            ],
            axis=2,
        )
        return np.clip(result, 0, 1)
    else:
        return np.clip(img + combined * grain_amp, 0, 1)
