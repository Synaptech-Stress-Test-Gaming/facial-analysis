"""Head movement as game input: tilt, nod, turn and lean, relative to the player's neutral posture.

stress_cam.py calls HeadMovement.update() once per frame and sends the result as the
"head_movement" object of every UDP message (docs/api.md).

Calibration runs alongside stress_cam's "relax, look at the screen" calibration (20 s by
default, restarted with c) and learns the neutral posture that all values are measured
from, so sitting normally reads 0:

  - the first second is skipped (the camera is still adjusting its exposure, and the
    player is still settling in);
  - only frames where the head is still count, so shifting in the chair or glancing
    at the keyboard doesn't move the zero;
  - the neutral posture is the median of those frames, robust to the odd outlier.

After calibration every frame gives, per movement:

  degrees / fraction  how far from neutral (lean: share of the neutral distance)
  amount              the same scaled to -1..1 (full_scale gives 1), easy to map to a control
  state               a word with hysteresis, so it doesn't flicker around the threshold

Directions are from the player's point of view in stress_cam's mirrored (selfie) image:

  tilt   ear toward a shoulder   left (-) / right (+)
  nod    chin up or down         down (-) / up (+)
  turn   look left or right      left (-) / right (+)
  lean   toward or away          back (-) / forward (+), i.e. closer to the screen

With --video input the image isn't mirrored, so left and right swap.

The rotation is measured from the neutral head pose, not by subtracting camera angles:
the head's current rotation relative to its neutral rotation is split into rotations
about the head's own axes. So neutral is exactly 0, and turning the head doesn't show up
as tilt even when the camera sits above or beside the screen. Rotations come from
MediaPipe's head pose matrix (Face.transform), the distance from its translation (cm).
Each value is smoothed with a 1 Euro filter, which hides landmark jitter while the head
is still but adds almost no lag when it moves.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from one_euro import OneEuroFilter


@dataclass(frozen=True)
class Movement:
    name: str
    unit: str          # "degrees", or "fraction" of the neutral distance
    negative: str      # state name below zero
    positive: str      # state name above zero
    full_scale: float  # this far from neutral gives amount -1 or 1
    enter: float       # the state switches on beyond this ...
    leave: float       # ... and back to "none" only below this (hysteresis)
    beta: float        # 1 Euro filter: how fast smoothing drops as the movement speeds up


MOVEMENTS = (
    Movement("tilt", "degrees", "left", "right", 30.0, 12.0, 8.0, 0.05),
    Movement("nod", "degrees", "down", "up", 25.0, 12.0, 8.0, 0.05),
    Movement("turn", "degrees", "left", "right", 35.0, 15.0, 10.0, 0.05),
    Movement("lean", "fraction", "back", "forward", 0.25, 0.10, 0.06, 5.0),
)
MIN_CUTOFF_HZ = 2.0      # smoothing while still; higher reacts faster but jitters more
SETTLE_SECONDS = 1.0     # skipped at the start of calibration
STILL_DEG_PER_S = 15.0   # calibration only uses frames where the head turns slower than this,
STILL_WINDOW = 0.25      # measured over this many seconds (longer than landmark jitter)
MAX_GAP_SECONDS = 0.2    # longer gaps between frames don't count as calibration time


def _pose(face) -> tuple[np.ndarray, float] | None:
    """The head's rotation (a proper 3x3 rotation matrix) and distance in cm, or None."""
    m = None if face is None else face.transform
    if m is None or np.shape(m) != (4, 4) or not np.isfinite(m).all():
        return None
    u, _, vt = np.linalg.svd(np.asarray(m, dtype=float)[:3, :3])  # nearest pure rotation
    if np.linalg.det(u @ vt) < 0:
        u[:, -1] *= -1
    distance = float(np.linalg.norm(m[:3, 3]))
    return (u @ vt, distance) if distance > 0 else None


def _degrees_between(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip((np.trace(a.T @ b) - 1) / 2, -1.0, 1.0))))


class HeadMovement:
    def __init__(self, calibration: float = 20.0):
        if not (math.isfinite(calibration) and calibration > 0):
            raise ValueError("Head calibration length must be finite and positive")
        self.calibration = calibration
        self.reset()

    def reset(self) -> None:
        """Forgets the neutral posture and learns it again from the next frames."""
        self.reference: tuple[np.ndarray, float] | None = None  # neutral rotation, distance
        self._rotations: list[np.ndarray] = []
        self._distances: list[float] = []
        self._started = None    # first face frame since the reset
        self._collected = 0.0   # seconds of still, settled face so far
        self._last_t = None
        self._recent: deque = deque()  # (t, rotation) over the last STILL_WINDOW seconds
        self._restart_tracking()

    @property
    def calibrated(self) -> bool:
        return self.reference is not None

    def update(self, t: float, face) -> dict:
        """One frame in (face may be None), the "head_movement" message object out."""
        pose = _pose(face)
        previous, self._last_t = self._last_t, (t if pose is not None else None)
        if pose is None:
            self._recent.clear()
            self._restart_tracking()  # after a gap, start fresh rather than from stale values
            return self._result(None)
        rotation, distance = pose
        if self.reference is None:
            self._calibrate(t, previous, rotation, distance)
            return self._result(None)
        neutral, neutral_distance = self.reference
        # Rotation since neutral, in the head's own axes: x across, y up, z out of the face.
        pitch, yaw, roll = cv2.RQDecomp3x3(neutral.T @ rotation)[0]
        values = {"tilt": -roll, "nod": -pitch, "turn": yaw,  # see the sign table above
                  "lean": (neutral_distance - distance) / neutral_distance}  # > 0: closer
        return self._result({m.name: self._filters[m.name](t, values[m.name]) for m in MOVEMENTS})

    def _calibrate(self, t: float, previous: float | None, rotation: np.ndarray, distance: float) -> None:
        if self._started is None:
            self._started = t
        self._recent.append((t, rotation))
        while t - self._recent[0][0] > STILL_WINDOW:
            self._recent.popleft()
        if t - self._started < SETTLE_SECONDS:
            return
        old_t, old_rotation = self._recent[0]
        if t - old_t < 0.1 or _degrees_between(old_rotation, rotation) / (t - old_t) > STILL_DEG_PER_S:
            return  # too little history to tell, or the head is moving
        if previous is not None and 0 < t - previous <= MAX_GAP_SECONDS:
            self._collected += t - previous
        self._rotations.append(rotation)
        self._distances.append(distance)
        if self._collected >= self.calibration:
            # Median rotation: median of the small rotations from the first sample.
            first = self._rotations[0]
            offsets = np.array([cv2.Rodrigues(first.T @ r)[0].ravel() for r in self._rotations])
            neutral = first @ cv2.Rodrigues(np.median(offsets, axis=0))[0]
            self.reference = (neutral, float(np.median(self._distances)))
            self._rotations, self._distances = [], []

    def _restart_tracking(self) -> None:
        self._filters = {m.name: OneEuroFilter(MIN_CUTOFF_HZ, m.beta) for m in MOVEMENTS}
        self._states = {m.name: "none" for m in MOVEMENTS}

    def _state(self, m: Movement, value: float) -> str:
        side = m.positive if value > 0 else m.negative
        if abs(value) >= m.enter or (abs(value) >= m.leave and self._states[m.name] == side):
            state = side
        else:
            state = "none"
        self._states[m.name] = state
        return state

    def _result(self, values: dict | None) -> dict:
        progress = 1.0 if self.calibrated else min(self._collected / self.calibration, 1.0)
        out = {"calibrated": self.calibrated, "calibration_progress": round(progress, 3)}
        for m in MOVEMENTS:
            if values is None:
                out[m.name] = None
                continue
            value = float(values[m.name])
            entry = {m.unit: round(value, 1 if m.unit == "degrees" else 3),
                     "amount": round(float(np.clip(value / m.full_scale, -1.0, 1.0)), 3),
                     "state": self._state(m, value)}
            if m.name == "lean":
                entry["cm"] = round(value * self.reference[1], 1)
            out[m.name] = entry
        return out
