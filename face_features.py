"""Face -> stress-related features, using Google's MediaPipe Face Landmarker.

For every webcam frame MediaPipe returns 478 face landmarks plus 52 "blendshape"
scores between 0 and 1 (browDownLeft, mouthPressRight, eyeBlinkLeft, ...).
Blendshapes are close to the facial Action Units (AUs) psychologists use to
describe expressions, which makes them a better input for stress than generic
emotion labels such as "happy" or "angry".

FeatureExtractor turns that stream of frames into one feature vector (a dict)
describing the last few seconds: average expression, blink rate, head movement...
"""

from __future__ import annotations

import math
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")

# Expression features and the MediaPipe blendshapes each one averages.
EXPRESSIONS = {
    "brow_down": ("browDownLeft", "browDownRight"),              # AU4 furrowed brow: effort, frustration
    "brow_inner_up": ("browInnerUp",),                           # AU1 worried brow
    "eye_squint": ("eyeSquintLeft", "eyeSquintRight"),           # AU7 tightened eyelids
    "eye_wide": ("eyeWideLeft", "eyeWideRight"),                 # AU5 eyes wide open: alarm, startle
    "nose_sneer": ("noseSneerLeft", "noseSneerRight"),           # AU9 nose wrinkle: annoyance
    "lip_press": ("mouthPressLeft", "mouthPressRight"),          # AU24 lips pressed together
    "lip_roll": ("mouthRollLower", "mouthRollUpper"),            # AU28 lips sucked in / bitten
    "mouth_frown": ("mouthFrownLeft", "mouthFrownRight"),        # AU15 mouth corners down
    "mouth_stretch": ("mouthStretchLeft", "mouthStretchRight"),  # AU20 lips pulled sideways: fear
    "chin_raise": ("mouthShrugLower",),                          # AU17 chin pushed up: tension, doubt
    "jaw_open": ("jawOpen",),                                    # talking, yelling, surprise
    "smile": ("mouthSmileLeft", "mouthSmileRight"),              # AU12 smile
}

# --- "Micro-tension" stress cues from fine mouth/jaw movements -----------------
# Every threshold for the cues below lives in this one block: durations in
# seconds, everything else a raw blendshape score (0-1).
CUE_NAMES = ("teeth_bare", "jaw_clench")  # grows as more cues are added
CUE_PERSIST_SECONDS = 0.6         # a cue must stay elevated this long before it contributes
CUE_ACTIVE_THRESHOLD = 0.15       # raw cue value above this counts as "elevated"
SPEECH_WINDOW_SECONDS = 1.0       # recent window used to detect talking/chewing
SPEECH_JAW_VAR_THRESHOLD = 0.01   # jawOpen variance above this = talking/chewing: suppress all cues

FEATURE_NAMES = (*EXPRESSIONS, *CUE_NAMES, "blink_rate", "head_motion", "face_size", "heart_rate")

# A few landmark indices of MediaPipe's 478-point face mesh.
NOSE_TIP, FACE_LEFT_EDGE, FACE_RIGHT_EDGE = 1, 234, 454


class CueTracker:
    """Computes the cues in CUE_NAMES from raw per-frame blendshapes.

    Each cue only turns on after CUE_PERSIST_SECONDS of continuous elevation
    (so a quick expression doesn't count), and all cues are suppressed while
    jawOpen is varying fast enough to mean talking or chewing.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._since = dict.fromkeys(CUE_NAMES)  # cue name -> t it became elevated, or None
        self._jaw_history = deque()              # (t, jawOpen) over the last SPEECH_WINDOW_SECONDS
        self.debug = {"inputs": {}, "raw": dict.fromkeys(CUE_NAMES, 0.0),
                      "gated": dict.fromkeys(CUE_NAMES, 0.0), "speaking": False}

    def update(self, t: float, blendshapes: dict) -> dict[str, float]:
        def b(name):
            return blendshapes.get(name, 0.0)

        jaw_open = b("jawOpen")
        self._jaw_history.append((t, jaw_open))
        while self._jaw_history and t - self._jaw_history[0][0] > SPEECH_WINDOW_SECONDS:
            self._jaw_history.popleft()
        speaking = (len(self._jaw_history) >= 5
                    and np.var([j for _, j in self._jaw_history]) > SPEECH_JAW_VAR_THRESHOLD)

        smile = (b("mouthSmileLeft") + b("mouthSmileRight")) / 2
        inputs = {"mouthUpperUpLeft": b("mouthUpperUpLeft"), "mouthUpperUpRight": b("mouthUpperUpRight"),
                  "mouthLowerDownLeft": b("mouthLowerDownLeft"), "mouthLowerDownRight": b("mouthLowerDownRight"),
                  "mouthStretchLeft": b("mouthStretchLeft"), "mouthStretchRight": b("mouthStretchRight"),
                  "mouthSmileLeft": b("mouthSmileLeft"), "mouthSmileRight": b("mouthSmileRight"),
                  "jawOpen": jaw_open, "mouthPressLeft": b("mouthPressLeft"), "mouthPressRight": b("mouthPressRight"),
                  "mouthShrugLower": b("mouthShrugLower"),
                  "mouthRollLower": b("mouthRollLower"), "mouthRollUpper": b("mouthRollUpper")}

        raw = {
            "teeth_bare": float(np.mean([inputs["mouthUpperUpLeft"], inputs["mouthUpperUpRight"],
                                          inputs["mouthLowerDownLeft"], inputs["mouthLowerDownRight"],
                                          inputs["mouthStretchLeft"], inputs["mouthStretchRight"]])
                                 * (1 - smile)),
            "jaw_clench": float((1 - jaw_open) * np.mean([inputs["mouthPressLeft"], inputs["mouthPressRight"],
                                                           inputs["mouthShrugLower"],
                                                           inputs["mouthRollLower"], inputs["mouthRollUpper"]])),
        }

        gated = dict.fromkeys(CUE_NAMES, 0.0)
        if speaking:
            self._since = dict.fromkeys(CUE_NAMES)
        else:
            for name in CUE_NAMES:
                if raw[name] > CUE_ACTIVE_THRESHOLD:
                    if self._since[name] is None:
                        self._since[name] = t
                else:
                    self._since[name] = None
                if self._since[name] is not None and t - self._since[name] >= CUE_PERSIST_SECONDS:
                    gated[name] = raw[name]

        self.debug = {"inputs": inputs, "raw": raw, "gated": gated, "speaking": speaking}
        return gated


@dataclass
class Face:
    points: np.ndarray  # (478, 2) landmark positions in pixels
    blendshapes: dict   # blendshape name -> score between 0 and 1

    @property
    def width(self) -> float:
        return float(np.linalg.norm(self.points[FACE_RIGHT_EDGE] - self.points[FACE_LEFT_EDGE]))

    def box(self) -> tuple[int, int, int, int]:
        (x0, y0), (x1, y1) = self.points.min(axis=0), self.points.max(axis=0)
        return int(x0), int(y0), int(x1), int(y1)


def ensure_model(path: Path) -> Path:
    """Downloads MediaPipe's face landmarker model (~3.7 MB) on first use."""
    if not path.exists():
        print(f"Downloading the face model to {path} ...")
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, partial)
        partial.replace(path)
    return path


class FaceDetector:
    """MediaPipe Face Landmarker in video mode, following one face."""

    def __init__(self, model_path: Path):
        options = vision.FaceLandmarkerOptions(
            # The model is passed as bytes because MediaPipe can't open file paths
            # with non-ASCII characters on Windows (e.g. folders named in Korean).
            base_options=BaseOptions(model_asset_buffer=Path(model_path).read_bytes()),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ms = -1

    def detect(self, frame_bgr: np.ndarray, t: float) -> Face | None:
        ms = max(int(t * 1000), self._last_ms + 1)  # MediaPipe needs increasing timestamps
        self._last_ms = ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        result = self._landmarker.detect_for_video(image, ms)
        if not result.face_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        points = np.array([(p.x * w, p.y * h) for p in result.face_landmarks[0]], dtype=np.float32)
        blendshapes = {c.category_name: c.score for c in result.face_blendshapes[0]}
        return Face(points, blendshapes)

    def close(self) -> None:
        self._landmarker.close()


class FeatureExtractor:
    """Summarizes the last few seconds of face observations as one feature vector.

    expressions  mean blendshape scores over the last `window` seconds
    blink_rate   blinks per minute over the last `blink_window` seconds of visible face
    head_motion  head speed in face-widths per second, mean over `window` (fidgeting)
    face_size    face width / image width; grows when you lean towards the screen
    heart_rate   webcam pulse in bpm from pulse.PulseEstimator, NaN when unreliable
    """

    def __init__(self, window: float = 2.0, blink_window: float = 30.0, pulse=None):
        self.window = window
        self.blink_window = blink_window
        self.pulse = pulse
        self.cues = CueTracker()
        self._frames = deque()   # (t, expression vector, head speed, face size, cue vector)
        self._visible = deque()  # (t, seconds of visible face this frame adds)
        self._blinks = deque()   # times of completed blinks
        self._eyes_open = None   # resting eyeBlink score with open eyes (differs per person)
        self._closed_at = None   # start of the current eye closure
        self._prev = None        # (t, nose position) of the previous frame with a face
        self._face_present = False

    def update(self, t: float, frame: np.ndarray, face: Face | None) -> None:
        self._forget_before(t)
        self._face_present = face is not None
        if face is None:
            self._prev = self._closed_at = None
            self.cues.reset()
            return

        expr = np.array([np.mean([face.blendshapes.get(n, 0.0) for n in names])
                         for names in EXPRESSIONS.values()])
        cue_values = self.cues.update(t, face.blendshapes)
        cues = np.array([cue_values[n] for n in CUE_NAMES])
        width = max(face.width, 1.0)
        nose = face.points[NOSE_TIP]
        speed = math.nan
        if self._prev is not None and 0 < t - self._prev[0] < 0.5:
            dt = t - self._prev[0]
            nose = 0.5 * nose + 0.5 * self._prev[1]  # light smoothing against landmark jitter
            speed = float(np.linalg.norm(nose - self._prev[1])) / width / dt
            self._visible.append((t, dt))
        self._prev = (t, nose)
        self._frames.append((t, expr, speed, width / frame.shape[1], cues))

        self._track_blinks(t, face)
        if self.pulse is not None:
            self.pulse.update(t, frame, face)

    def features(self, t: float) -> dict | None:
        """Feature vector for the last few seconds, or None if no face was seen recently."""
        self._forget_before(t)
        if len(self._frames) < 5:
            return None
        feats = dict(zip(EXPRESSIONS, np.mean([f[1] for f in self._frames], axis=0).tolist()))
        feats.update(zip(CUE_NAMES, np.mean([f[4] for f in self._frames], axis=0).tolist()))
        if not self._face_present:  # no stale cues once the face is gone, even if old frames linger
            for name in CUE_NAMES:
                feats[name] = 0.0
        speeds = [f[2] for f in self._frames if not math.isnan(f[2])]
        feats["head_motion"] = float(np.mean(speeds)) if speeds else math.nan
        feats["face_size"] = float(np.mean([f[3] for f in self._frames]))
        seen = sum(dt for _, dt in self._visible)
        feats["blink_rate"] = 60.0 * len(self._blinks) / seen if seen >= 10 else math.nan
        feats["heart_rate"] = self.pulse.current(t) if self.pulse is not None else math.nan
        return feats

    def _track_blinks(self, t: float, face: Face) -> None:
        score = (face.blendshapes.get("eyeBlinkLeft", 0.0) + face.blendshapes.get("eyeBlinkRight", 0.0)) / 2
        if self._eyes_open is None:
            self._eyes_open = score
        closed_above = self._eyes_open + 0.25
        if self._closed_at is None:
            if score > closed_above:
                self._closed_at = t
            else:
                self._eyes_open += 0.05 * (score - self._eyes_open)
        elif score < closed_above - 0.12:
            if t - self._closed_at < 0.5:  # longer closures are not blinks
                self._blinks.append(t)
            self._closed_at = None
        elif t - self._closed_at > 1.0:  # "closed" for long: probably looking down, re-learn
            self._eyes_open, self._closed_at = score, None

    def _forget_before(self, t: float) -> None:
        for queue, horizon in ((self._frames, self.window), (self._visible, self.blink_window)):
            while queue and t - queue[0][0] > horizon:
                queue.popleft()
        while self._blinks and t - self._blinks[0] > self.blink_window:
            self._blinks.popleft()
