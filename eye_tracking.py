"""Image-relative eye geometry and time-weighted visual orientation."""
from collections import deque
from dataclasses import dataclass
from math import isfinite

import numpy as np

# Both sets are ordered from image-left to image-right.
EYES = ((33, 160, 158, 133, 153, 144, 468),
        (362, 385, 387, 263, 373, 380, 473))


@dataclass(frozen=True)
class EyeSample:
    x: float
    y: float
    ear: float


def measure(points):
    if points is None or points.ndim != 2 or points.shape[0] < 474 or points.shape[1] != 2:
        return None
    eyes = []
    for indices in EYES:
        left, upper1, upper2, right, lower2, lower1, iris = points[list(indices)]
        if not np.isfinite(points[list(indices)]).all():
            return None
        horizontal = right - left
        width = float(np.linalg.norm(horizontal))
        if width < 2:
            return None
        axis = horizontal / width
        down = np.array([-axis[1], axis[0]])
        offset = iris - (left + right) / 2
        gap = np.linalg.norm(upper1 - lower1) + np.linalg.norm(upper2 - lower2)
        eyes.append((float(offset @ axis) / width, float(offset @ down) / width,
                     float(gap) / (2 * width)))
    return EyeSample(*map(float, np.mean(eyes, axis=0)))


class EyeTracker:
    max_gap = 0.20

    def __init__(self, calibration=3.0, window=3.0):
        if not all(isfinite(v) and v > 0 for v in (calibration, window)):
            raise ValueError("Eye calibration and window lengths must be finite and positive")
        self.calibration = calibration
        self.window = window
        self.reset()

    def reset(self):
        self.reference = None
        self.reference_samples = []
        self.reference_time = 0.0
        self.previous = None
        self.closed_at = None
        self.stable_since = None
        self.stable_anchor = None
        self.orientations = deque()
        self.visible = deque()
        self.blinks = deque()

    def update(self, t, points=None, pose=None, failure=False):
        sample = measure(points)
        status = "tracking_failure" if failure or (points is not None and sample is None) else "missing_face"
        if pose is not None and (len(pose) != 3 or not all(isfinite(value) for value in pose)):
            pose = None
        previous = self.previous
        self.previous = (t, sample, pose) if sample is not None else None
        dt = t - previous[0] if previous else 0.0
        if not 0 < dt <= self.max_gap or sample is None:
            dt = 0.0
            self.closed_at = None
            self.stable_since = self.stable_anchor = None
            if self.reference is None:
                self.reference_samples.clear()
                self.reference_time = 0.0

        if self.reference is None and dt and (pose is None or previous[2] is None):
            self.reference_samples.clear()
            self.reference_time = 0.0
        if (sample and self.reference is None and dt and pose is not None and
                previous[2] is not None and sample.ear >= 0.16 and previous[1].ear >= 0.16):
            self.reference_time += dt
            self.reference_samples.append((sample.x, sample.y, sample.ear, pose))
            if self.reference_time >= self.calibration:
                x, y, ear = np.median([s[:3] for s in self.reference_samples], axis=0)
                poses = [s[3] for s in self.reference_samples]
                angles = tuple(map(float, np.median(poses, axis=0)))
                self.reference = (float(x), float(y), float(ear), angles)
                self.reference_samples.clear()

        direction = centered = closure = fixation = dx = dy = None
        if sample:
            status = "uncertain"
        if sample and self.reference:
            x, y, ear, reference_pose = self.reference
            threshold = ear * (0.68 if self.closed_at is not None else 0.60)
            if sample.ear < threshold:
                if self.closed_at is None:
                    self.closed_at = t
                closure = t - self.closed_at
            elif self.closed_at is not None:
                duration = t - self.closed_at
                if 0.06 <= duration <= 0.45:
                    self.blinks.append(t)
                self.closed_at = None
            if dt:
                self.visible.append((t, dt))

            if closure is None:
                dx, dy = sample.x - x, sample.y - y
                direction = "center"
                if abs(dx) > 0.07 or abs(dy) > 0.055:
                    if abs(dx) / 0.07 >= abs(dy) / 0.055:
                        direction = "left" if dx < 0 else "right"
                    else:
                        direction = "up" if dy < 0 else "down"
                if pose is not None:
                    centered = direction == "center" and all(
                        abs((a - b + 180) % 360 - 180) <= 15
                        for a, b in zip(pose[:2], reference_pose[:2])
                    )
                    status = "valid"
                    if dt and self.orientations and self.orientations[-1][0] == previous[0]:
                        weight = dt
                    else:
                        weight = 0.0
                    self.orientations.append((t, weight, centered, dx, dy))
                    if self.stable_anchor is None or max(
                        abs(dx - self.stable_anchor[0]), abs(dy - self.stable_anchor[1])
                    ) > 0.035:
                        self.stable_anchor = (dx, dy)
                        self.stable_since = t
                    fixation = t - self.stable_since

        if status != "valid":
            self.stable_anchor = self.stable_since = None
        for queue, horizon in ((self.orientations, self.window), (self.visible, 30.0)):
            while queue and t - queue[0][0] >= horizon:
                queue.popleft()
        while self.blinks and t - self.blinks[0] >= 30:
            self.blinks.popleft()
        weighted = [(min(row[1], self.window - (t - row[0])), *row[2:])
                    for row in self.orientations if row[1]]
        coverage = sum(row[0] for row in weighted)
        score = dispersion = None
        if status == "valid" and coverage >= self.window * 0.5:
            score = sum(weight for weight, center, _, _ in weighted if center) / coverage
            xy = np.asarray([row[2:] for row in weighted])
            weights = [row[0] for row in weighted]
            mean = np.average(xy, axis=0, weights=weights)
            dispersion = float(np.sqrt(np.average(np.sum((xy - mean) ** 2, axis=1), weights=weights)))
        seen = sum(min(span, 30 - (t - end)) for end, span in self.visible)
        return {
            "tracking_state": status,
            "tracking_valid": status == "valid",
            "gaze": {
                "method": "center_reference",
                "direction": direction,
                "relative_x": sample.x if sample and closure is None else None,
                "relative_y": sample.y if sample and closure is None else None,
                "center_offset_x": dx, "center_offset_y": dy,
                "screen_x": None, "screen_y": None, "on_screen": None,
                "center_oriented": centered, "dispersion": dispersion,
                "stable_seconds": fixation,
            },
            "eyes": {
                "ear": sample.ear if sample else None,
                "closure_seconds": closure,
                "blink_rate_per_minute": 60 * len(self.blinks) / seen if sample and seen >= 10 else None,
            },
            "head_pose": dict(zip(("pitch_deg", "yaw_deg", "roll_deg"), pose)) if pose is not None else None,
            "attention": {
                "score": score, "coverage": min(1.0, coverage / self.window),
                "method": "center_orientation_fraction",
            },
            "eye_calibration_progress": min(1.0, self.reference_time / self.calibration),
        }
