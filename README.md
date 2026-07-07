# film sim

Physics-based film stock simulator for photos and video. Upload an image or video clip, pick a film stock, and get back a result that models the actual photochemistry — not a LUT or a Lightroom preset.

---

## What makes it different

Most "film look" tools apply a color grade and call it done. This models the process:

| Component | What it simulates |
|-----------|-------------------|
| **H&D curve** | Hurter–Driffield characteristic curve of each emulsion — toe, shoulder, and gain tuned per stock. Operates in linear light, not log space. |
| **FFT grain** | Band-limited spatially correlated noise via FFT filtering. Not `np.random.normal()` — grain has a frequency signature. Each color channel sits at a different spatial scale, and the R/B layers are offset 1px to simulate dye cloud misalignment between emulsion layers. |
| **Halation** | Light bleeds back through the film base from the support layer. Cinestill 800T gets radius=35px / strength=0.18 almost entirely in the red channel (it's motion picture film with the remjet backing removed). |
| **Panchromatic B&W** | HP5 uses 0.25R + 0.68G + 0.07B — the actual spectral sensitivity weighting of a panchromatic emulsion — not Rec.709 luminance. |
| **Auto color grade (video)** | Two-pass pipeline: sample ~40 frames → compute global WB, exposure, and per-channel levels → apply consistently to every frame. Avoids the per-frame flicker of naive auto-grade. Gray-world WB is clamped to ±30% so intentionally warm/cool scenes aren't crushed. |

---

## Film stocks

| Stock | Type | Character |
|-------|------|-----------|
| Portra 200 | Color negative | Warm skin tones, soft highlights, fine grain |
| Portra 800 | Color negative | Warmer, lifted blacks, chunky grain |
| Ektar 100 | Color negative | Most saturated negative film ever made |
| Cinestill 800T | Color negative | Tungsten-balanced, teal shadows, signature red halation |
| Velvia 50 | Slide (reversal) | Electric saturation, inky blacks, zero base lift |
| HP5 @ 1600 | B&W negative | Pushed 2 stops, hard shadow blocking, split tone |

---

## Stack

- **Python / Flask** — backend and processing pipeline  
- **NumPy / SciPy** — all image math (FFT grain, Gaussian halation, H&D curves)  
- **Pillow** — image I/O  
- **OpenCV** — video frame I/O  
- Vanilla HTML/CSS/JS frontend — no framework, single file

---

## Run locally

```bash
git clone https://github.com/YOUR_USERNAME/filmsim
cd filmsim
pip install -r requirements.txt
python app.py
```

Open [http://localhost:5000](http://localhost:5000).

**Photo:** upload any JPEG/PNG/WEBP, pick a stock, click develop.  
**Video:** switch to the video tab, upload a clip (mp4, mov, mkv, avi, etc.), pick a stock, click develop. Processing runs in the background — a progress bar tracks frame completion. Output is a downloadable `.mp4`. Note: audio is not preserved.

---

## Project structure

```
filmsim/
├── app.py               # Flask routes (photo + video)
├── requirements.txt
├── stocks/
│   ├── base.py          # Shared physics: H&D curve, FFT grain, halation
│   ├── portra200.py
│   ├── portra800.py
│   ├── ektar100.py
│   ├── cinestill800t.py
│   ├── velvia50.py
│   └── hp5_1600.py
├── video/
│   ├── auto_grade.py    # Scene analysis + global grade computation
│   └── processor.py     # Two-pass video pipeline
└── templates/
    └── index.html       # Single-page UI
```

---

## How the grain works

```python
# Band-limited noise via FFT — grain has a physical frequency signature
def shaped_noise(h, w, sigma, seed):
    noise = rng.standard_normal((h, w))
    freq_mag = sqrt(freq_x² + freq_y²)
    envelope = exp(-freq_mag² / (2 * (sigma / max(h,w))²))
    return real(ifft2(fft2(noise) * envelope))

# R/B layers offset 1px — dye clouds at different emulsion depths don't align
grain_r = roll(roll(grain_r,  1, axis=0),  1, axis=1)
grain_b = roll(roll(grain_b, -1, axis=0), -1, axis=1)
```

## How Cinestill halation works

```python
# Highlights bleed back almost entirely in the red channel
# (remjet backing removed → light reflects off film base → red dye layer takes the hit)
halo_r = gaussian_filter(highlights[:,:,0], sigma=35)
halo_g = gaussian_filter(highlights[:,:,1], sigma=17.5)   # half
halo_b = gaussian_filter(highlights[:,:,2], sigma=7)      # 0.2x
img += stack([halo_r, halo_g, halo_b]) * 0.18
```
