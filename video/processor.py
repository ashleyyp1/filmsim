import cv2
import importlib
import numpy as np

from stocks.base import srgb_to_linear, linear_to_srgb
from video.auto_grade import compute_grade, apply_grade

# Downscale long edge to this maximum for processing speed
MAX_DIM = 1280

STOCKS = {
    'portra200':    'stocks.portra200',
    'portra800':    'stocks.portra800',
    'ektar100':     'stocks.ektar100',
    'cinestill800t':'stocks.cinestill800t',
    'velvia50':     'stocks.velvia50',
    'hp5_1600':     'stocks.hp5_1600',
}


def _out_dims(w, h):
    """Scale to MAX_DIM on the long edge; keep values even (required for yuv420)."""
    if max(w, h) <= MAX_DIM:
        return w, h
    s = MAX_DIM / max(w, h)
    return (int(w * s) & ~1), (int(h * s) & ~1)


def process_video(input_path, output_path, stock_name, progress_cb=None):
    """
    Two-pass pipeline:
      1. Sample ~40 frames → compute a single consistent global grade.
      2. Process every frame: auto-grade → film stock → write output.

    Returns the first processed preview frame as an RGB uint8 array, or None.
    """
    if stock_name not in STOCKS:
        raise ValueError(f'Unknown stock: {stock_name}')

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError('Cannot open video file')

    fps     = cap.get(cv2.CAP_PROP_FPS) or 24.0
    orig_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    out_w, out_h = _out_dims(orig_w, orig_h)
    do_resize = (out_w != orig_w or out_h != orig_h)

    mod = importlib.import_module(STOCKS[stock_name])

    # ── Pass 1: sample frames for scene analysis ─────────────────────────
    if progress_cb:
        progress_cb(0.0, 'analyzing scene…')

    n_samples    = min(40, total)
    sample_idx   = set(np.linspace(0, total - 1, n_samples, dtype=int))
    samples      = []
    fi           = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        if fi in sample_idx:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            if do_resize:
                rgb = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
            samples.append(srgb_to_linear(rgb))
        fi += 1

    if not samples:
        cap.release()
        raise RuntimeError('Could not read any frames from this video')

    grade_params = compute_grade(samples)
    del samples

    # ── Pass 2: process every frame ───────────────────────────────────────
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError('Could not open VideoWriter — check codec support')

    preview_rgb8 = None
    fi = 0

    try:
        while True:
            ret, bgr = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            if do_resize:
                rgb = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)

            linear  = srgb_to_linear(rgb)
            graded  = apply_grade(linear, grade_params)
            result  = linear_to_srgb(mod.process(graded))
            rgb8    = (np.clip(result, 0, 1) * 255).astype(np.uint8)

            writer.write(cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR))

            # Capture a preview frame near the beginning
            if fi == min(12, total - 1):
                preview_rgb8 = rgb8.copy()

            fi += 1
            if progress_cb:
                progress_cb(fi / total, 'processing frames…')
    finally:
        cap.release()
        writer.release()

    return preview_rgb8
