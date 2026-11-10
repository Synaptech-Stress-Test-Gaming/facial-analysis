"""Experimental: heart rate from the webcam (remote photoplethysmography, rPPG).

Each heartbeat pushes blood into the face and changes skin color very slightly.
We average the color of both cheeks every frame, run the POS algorithm (Wang et
al., 2017, "Algorithmic Principles of Remote PPG") over the last ~12 seconds and
take the strongest frequency between 45 and 180 bpm.

It needs steady, bright light and a fairly still head. Light from a flashing game
screen and fast movement add noise, so the estimate is NaN whenever the pulse
peak is not clearly stronger than everything else in the signal.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, filtfilt

RATE = 30.0                 # color traces are resampled to this many samples per second
MIN_HZ, MAX_HZ = 0.75, 3.0  # 45 - 180 bpm

# (outer eye corner, mouth corner) landmark pairs, one per side of the face.
CHEEK_ANCHORS = ((33, 61), (263, 291))
NOSE_TIP = 1


class PulseEstimator:
    # min_quality 0.5: on pure camera noise the quality stays below ~0.5, while a real
    # pulse of 0.2% skin-color change already scores ~0.6-0.75.
    def __init__(self, window: float = 12.0, min_quality: float = 0.5):
        self.window = window
        self.min_quality = min_quality
        self.quality = 0.0  # share of the signal power at the pulse frequency, 0..1
        self.rois = []      # cheek boxes used in the last frame, for drawing
        self._times = deque()
        self._colors = deque()  # mean (R, G, B) of the cheeks per frame
        self._estimates = deque(maxlen=5)  # consecutive clean estimates (bpm), one per second
        self._last_estimate = -math.inf

    def current(self, t: float) -> float:
        """Heart rate in bpm, or NaN unless the last 3 estimates were clean and agree."""
        recent = list(self._estimates)[-3:]
        if len(recent) < 3 or max(recent) - min(recent) > 8 or t - self._times[-1] > 1.0:
            return math.nan
        return float(np.median(self._estimates))

    def update(self, t: float, frame: np.ndarray, face) -> None:
        if self._times and t - self._times[-1] > 0.5:  # face was lost: the trace is broken
            self.reset()
        self.rois = cheek_boxes(face, frame.shape)
        pixels = [frame[y0:y1, x0:x1].reshape(-1, 3) for x0, y0, x1, y1 in self.rois]
        if not pixels:
            return
        b, g, r = np.concatenate(pixels).mean(axis=0)
        self._times.append(t)
        self._colors.append((r, g, b))
        while t - self._times[0] > self.window:
            self._times.popleft()
            self._colors.popleft()
        if t - self._last_estimate >= 1.0 and t - self._times[0] >= 0.75 * self.window:
            self._last_estimate = t
            self._estimate()

    def reset(self) -> None:
        self._times.clear()
        self._colors.clear()
        self._estimates.clear()
        self.quality = 0.0

    def _estimate(self) -> None:
        times = np.array(self._times)
        grid = np.arange(times[0], times[-1], 1.0 / RATE)
        traces = np.array([np.interp(grid, times, channel) for channel in np.array(self._colors).T])
        signal = pos(traces, int(1.6 * RATE))
        b, a = butter(3, [MIN_HZ, MAX_HZ], btype="band", fs=RATE)
        signal = filtfilt(b, a, signal)

        n = 8 * len(signal)  # zero-padding gives a finer frequency grid
        freqs = np.fft.rfftfreq(n, 1.0 / RATE)
        power = np.abs(np.fft.rfft(signal * np.hanning(len(signal)), n)) ** 2
        band = (freqs >= MIN_HZ) & (freqs <= MAX_HZ)
        peak = freqs[band][np.argmax(power[band])]
        total = power[band].sum()
        self.quality = float(power[band & (np.abs(freqs - peak) <= 0.12)].sum() / total) if total > 0 else 0.0
        if self.quality >= self.min_quality:
            self._estimates.append(60.0 * peak)
        else:
            self._estimates.clear()


def pos(rgb: np.ndarray, win: int) -> np.ndarray:
    """POS algorithm: (3, n) R, G, B traces -> (n,) pulse signal."""
    windows = sliding_window_view(rgb, win, axis=1)               # (3, n - win + 1, win)
    c = windows / (windows.mean(axis=2, keepdims=True) + 1e-9)     # remove skin tone and brightness
    s1 = c[1] - c[2]                                               # G - B
    s2 = c[1] + c[2] - 2 * c[0]                                    # G + B - 2R
    h = s1 + s1.std(axis=1, keepdims=True) / (s2.std(axis=1, keepdims=True) + 1e-9) * s2
    h -= h.mean(axis=1, keepdims=True)
    out = np.zeros(rgb.shape[1])
    np.add.at(out, np.arange(h.shape[0])[:, None] + np.arange(win), h)  # overlap-add the windows
    return out


def cheek_boxes(face, shape) -> list[tuple[int, int, int, int]]:
    """Two square skin patches, one per cheek, as (x0, y0, x1, y1) pixel boxes."""
    p, half = face.points, 0.08 * face.width
    h, w = shape[:2]
    boxes = []
    for eye, mouth in CHEEK_ANCHORS:
        center = (p[eye] + p[mouth]) / 2
        center += 0.15 * (p[NOSE_TIP] - center)  # a bit towards the nose, away from the jaw line
        x0, y0 = np.maximum(center - half, 0).astype(int)
        x1, y1 = np.minimum(center + half, (w, h)).astype(int)
        if x1 > x0 and y1 > y0:
            boxes.append((int(x0), int(y0), int(x1), int(y1)))
    return boxes
