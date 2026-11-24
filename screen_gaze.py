"""Screen gaze: where on the screen are you looking? Webcam gaze with a dot calibration.

  python screen_gaze.py                 calibrate with red dots, then follow your gaze live
  python screen_gaze.py --points 16     more calibration dots: takes longer, a bit more accurate
  python screen_gaze.py --camera 1      another webcam (default 0, or the STRESS_CAMERA variable)

stress_cam.py's eye tracking (eye_tracking.py) calibrates on one centre target and reports
gaze direction relative to it; screen coordinates are left null there. This app fills
that gap as a separate program: it calibrates on several dots across the screen and
reports where on the screen you look.

The window goes full screen. Sit the way you normally do, keep your head still, and look
at each red dot until it disappears (about 25 s for 13 dots). The app then shows a dot
where you are looking and sends that point as JSON over UDP to 127.0.0.1:5007 (see
docs/api.md), next to stress_cam.py's 5005 and the EEG's 5006.

How it works: eye_tracking.measure() gives where each iris sits between the eye corners,
in eye widths, and how open the eyes are; MediaPipe's gaze blendshapes
(eyeLookUp/Down/In/Out) add a second opinion. While you look at each calibration dot
those values are recorded with the dot's position, and a ridge regression learns the
mapping from eye values to screen position.

Webcam gaze is coarse: expect an error of a few centimetres on the screen, enough for
"which part of the screen", not for single words. Moving your head after calibrating
makes it worse; the app warns you when that happens.

Keys: space start / continue, c calibrate again, g grid, p camera preview, q quit.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time

import cv2
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from eye_tracking import EYES, measure
from face_features import Face, FaceDetector, ensure_model
from one_euro import OneEuroFilter
from stress_cam import FACE_MODEL, UdpSender, open_camera

WINDOW = "Screen gaze"
GAZE_PORT = 5007
FEATURES = ("iris_x", "iris_y", "eye_open", "look_x", "look_y")

SETTLE_SECONDS = 0.9    # the dot shrinks while your eyes jump to it and settle (not recorded)
COLLECT_SECONDS = 1.0   # then eye features are recorded while you look at the small dot
MAX_WAIT_SECONDS = 4.0  # give up on a dot when the eyes aren't seen for this long
MIN_SAMPLES = 8         # usable frames a dot needs to count
NO_FACE_SECONDS = 8.0   # stop calibrating when the face has been gone this long
# "You moved since calibrating": head turned, face moved sideways (in face widths), or
# leaned in or out (relative change of the face width in the image).
HEAD_MOVED_DEGREES, HEAD_MOVED_SHIFT, HEAD_MOVED_SCALE = 8.0, 0.35, 0.15

# OpenCV colors are (blue, green, red).
BACKGROUND = (28, 28, 28)
WHITE, GRAY, DARK = (235, 235, 235), (150, 150, 150), (75, 75, 75)
RED, GREEN, YELLOW = (40, 40, 230), (90, 200, 90), (0, 200, 255)
HIGHLIGHT = (55, 85, 55)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def eye_features(face: Face) -> np.ndarray | None:
    """Where the eyes point, as [iris_x, iris_y, eye_open, look_x, look_y], or None.

    iris_x, iris_y, eye_open  eye_tracking.measure(): iris offset from the eye's middle and
                              lid gap, in eye widths, both eyes averaged
    look_x, look_y            MediaPipe's gaze blendshapes: right minus left, up minus down
    """
    sample = measure(face.points)
    if sample is None:
        return None
    bs = face.blendshapes.get
    look_x = (bs("eyeLookOutLeft", 0.0) - bs("eyeLookInLeft", 0.0)
              + bs("eyeLookInRight", 0.0) - bs("eyeLookOutRight", 0.0)) / 2
    look_y = (bs("eyeLookUpLeft", 0.0) + bs("eyeLookUpRight", 0.0)
              - bs("eyeLookDownLeft", 0.0) - bs("eyeLookDownRight", 0.0)) / 2
    return np.array([sample.x, sample.y, sample.ear, look_x, look_y])


def head_position(face: Face | None) -> np.ndarray | None:
    """[pitch, yaw] in degrees, then the face's centre x, y and width in image pixels."""
    if face is None or face.pose is None:
        return None
    x0, y0, x1, y1 = face.box()
    return np.array([face.pose[0], face.pose[1], (x0 + x1) / 2, (y0 + y1) / 2, max(face.width, 1.0)])


def head_moved(now: np.ndarray, ref: np.ndarray) -> bool:
    turned = max(abs((a - b + 180) % 360 - 180) for a, b in zip(now[:2], ref[:2]))
    shift = float(np.linalg.norm(now[2:4] - ref[2:4])) / ref[4]
    scale = abs(now[4] / ref[4] - 1)
    # bool(): comparisons of numpy numbers give numpy.bool_, which json can't encode
    return bool(turned > HEAD_MOVED_DEGREES or shift > HEAD_MOVED_SHIFT or scale > HEAD_MOVED_SCALE)


def calibration_targets(n: int, margin: float = 0.07) -> list[tuple[float, float]]:
    """Dot positions as fractions of the screen: a 3x3 or 4x4 grid (13 = 3x3 plus 4 in between).
    The first dot is in the middle; the rest come in random order, so you can't anticipate them."""
    lines = np.linspace(margin, 1 - margin, 4 if n == 16 else 3)
    points = [(float(x), float(y)) for y in lines for x in lines]
    if n == 13:
        between = ((lines[0] + lines[1]) / 2, (lines[1] + lines[2]) / 2)
        points += [(float(x), float(y)) for y in between for x in between]
    first = min(points, key=lambda p: (p[0] - 0.5) ** 2 + (p[1] - 0.5) ** 2)
    rest = [p for p in points if p != first]
    order = np.random.default_rng().permutation(len(rest))
    return [first, *(rest[i] for i in order)]


def clean(samples: list[np.ndarray]) -> np.ndarray:
    """Drops the frames of one dot that don't look like the rest: blinks, glances away, glitches."""
    if not samples:
        return np.empty((0, len(FEATURES)))
    x = np.array(samples)
    median = np.median(x, axis=0)
    spread = 1.4826 * np.median(np.abs(x - median), axis=0) + 1e-6
    keep = np.all(np.abs(x - median) <= 3.5 * spread, axis=1)
    keep &= x[:, 2] >= 0.6 * median[2]  # eyes half closed: a blink
    return x[keep]


class Calibration:
    """Shows the dots one after another and records eye features while you look at each."""

    def __init__(self, targets: list[tuple[float, float]], t: float):
        self.targets = targets
        self.samples: list[list[np.ndarray]] = [[] for _ in targets]
        self.heads: list[np.ndarray] = []
        self.queue = list(range(len(targets)))
        self.retried: set[int] = set()
        self.shown = 0
        self.t0 = t

    @property
    def done(self) -> bool:
        return not self.queue

    @property
    def current(self) -> int:
        return self.queue[0]

    @property
    def total(self) -> int:
        return len(self.targets) + len(self.retried)

    def settle(self) -> float:
        return SETTLE_SECONDS + (0.8 if self.shown == 0 else 0.0)  # extra time to find the first dot

    def update(self, t: float, feats: np.ndarray | None, head: np.ndarray | None) -> None:
        if self.done:
            return
        age, samples = t - self.t0, self.samples[self.current]
        if age >= self.settle() and feats is not None:
            samples.append(feats)
            if head is not None:
                self.heads.append(head)
        enough = age >= self.settle() + COLLECT_SECONDS and len(samples) >= 2 * MIN_SAMPLES
        if enough or age >= self.settle() + MAX_WAIT_SECONDS:
            index = self.queue.pop(0)
            if not enough and index not in self.retried:  # show this dot once more at the end
                self.retried.add(index)
                self.samples[index] = []
                self.queue.append(index)
            self.shown += 1
            self.t0 = t

    def progress(self, t: float) -> tuple[float, float]:
        """How far the current dot is through settling and through recording, each 0..1."""
        age = t - self.t0
        return min(age / self.settle(), 1.0), float(np.clip((age - self.settle()) / COLLECT_SECONDS, 0, 1))


class GazeModel:
    """Eye features -> screen position in pixels (ridge regression, optionally quadratic).

    The polynomial degree and the regularization are picked by leave-one-dot-out
    cross-validation: every dot is predicted by a model trained only on the other dots.
    The same predictions give the accuracy estimate, so it is measured on dots the model
    did not learn from.
    """

    CANDIDATES = [(degree, alpha) for degree in (1, 2) for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0)]

    def __init__(self):
        self.model = None

    def fit(self, x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
        best = None
        for degree, alpha in self.CANDIDATES:
            predicted = cross_val_predict(self._make(degree, alpha), x, y, groups=groups, cv=LeaveOneGroupOut())
            dots = []
            for g in np.unique(groups):
                m = groups == g
                dots.append((y[m][0], predicted[m].mean(axis=0),
                             float(np.linalg.norm(predicted[m] - y[m], axis=1).mean())))
            error = float(np.mean([d[2] for d in dots]))
            if best is None or error < best["error"]:
                best = {"error": error, "degree": degree, "alpha": alpha, "dots": dots}
        self.model = self._make(best["degree"], best["alpha"]).fit(x, y)
        return best

    def predict(self, feats: np.ndarray) -> np.ndarray:
        return self.model.predict(feats[None])[0]

    @staticmethod
    def _make(degree: int, alpha: float):
        return make_pipeline(StandardScaler(), PolynomialFeatures(degree, include_bias=False), Ridge(alpha=alpha))


class ScreenGazeTracker:
    """The app's state: intro -> calibrating -> result -> tracking (c goes back to calibrating)."""

    def __init__(self, args: argparse.Namespace, size: tuple[int, int]):
        self.args = args
        self.w, self.h = size
        self.s = self.h / 900  # text scale
        self.mode = "intro"
        self.calibration: Calibration | None = None
        self.model: GazeModel | None = None
        self.report: dict | None = None
        self.frame = self.face = self.gaze = None
        self.raw_gaze = None   # unsmoothed, unclipped prediction, for on_screen
        self.timestamp = None  # wall-clock time of the current frame
        self.last_face = 0.0   # when the face was last seen
        self.blink_open = 0.0  # eye_open below this counts as a blink
        self.head_ref, self.head_moved = None, False
        self.filter = OneEuroFilter(min_cutoff=args.smoothing)
        self.show_grid, self.show_camera = True, True
        self.udp = None if args.udp == "off" else UdpSender(args.udp, rate=30.0)

    # --- state ---------------------------------------------------------------------------

    def step(self, t: float, frame: np.ndarray, face: Face | None, timestamp: float | None = None) -> None:
        self.frame, self.face, self.timestamp = frame, face, timestamp
        feats = eye_features(face) if face is not None else None
        head = head_position(face)
        if face is not None:
            self.last_face = t
        if self.mode == "calibrating":
            self.calibration.update(t, feats, head)
            if t - self.last_face > NO_FACE_SECONDS:
                self.calibration.queue.clear()  # give up; the result screen says why
            if self.calibration.done:
                self._fit()
        elif self.mode == "tracking":
            if feats is not None and feats[2] >= self.blink_open:  # during a blink: keep the last point
                self.raw_gaze = self.model.predict(feats)
                self.gaze = self.filter(t, self.raw_gaze)
            if head is not None and self.head_ref is not None:
                self.head_moved = head_moved(head, self.head_ref)
        if self.udp:
            self.udp.send(t, self.message(t))

    def key(self, k: int, t: float) -> bool:
        """Handles a key press; False means quit."""
        if k in (ord("q"), 27):  # 27 = Esc
            return False
        if k == ord(" ") and self.mode == "intro" or k == ord("c"):
            self.calibration = Calibration(calibration_targets(self.args.points), t)
            self.mode, self.gaze, self.raw_gaze, self.last_face = "calibrating", None, None, t
        elif k == ord(" ") and self.mode == "result" and self.model is not None:
            self.mode = "tracking"
            self.filter.reset()
        elif k == ord("g"):
            self.show_grid = not self.show_grid
        elif k == ord("p"):
            self.show_camera = not self.show_camera
        return True

    def _fit(self) -> None:
        cal = self.calibration
        x, y, groups, eyes_open = [], [], [], []
        for i, samples in enumerate(cal.samples):
            kept = clean(samples)
            if len(kept) < MIN_SAMPLES:
                continue
            x.append(kept)
            y += [(cal.targets[i][0] * self.w, cal.targets[i][1] * self.h)] * len(kept)
            groups += [i] * len(kept)
            eyes_open.append(float(np.median(kept[:, 2])))
        self.mode, self.model = "result", None
        usable = len(eyes_open)
        if usable < max(6, len(cal.targets) // 2):
            self.report = {"ok": False, "usable": usable, "total": len(cal.targets)}
            print(f"Calibration failed: only {usable} of {len(cal.targets)} dots had a clear view of your eyes.")
            return
        self.model = GazeModel()
        fit = self.model.fit(np.vstack(x), np.array(y), np.array(groups))
        self.report = {"ok": True, "usable": usable, "total": len(cal.targets), **fit}
        self.blink_open = 0.6 * min(eyes_open)  # more closed than when looking at the lowest dot
        self.head_ref = np.median(cal.heads, axis=0) if cal.heads else None
        self.head_moved = False
        print(f"Calibration: {usable} of {len(cal.targets)} dots used, accuracy about {fit['error']:.0f} px "
              f"({100 * fit['error'] / self.w:.0f}% of the screen width; "
              f"{'linear' if fit['degree'] == 1 else 'quadratic'} model).")

    def message(self, t: float) -> dict:
        """What a game receives on port 5007 (documented in docs/api.md)."""
        on = self.mode == "tracking" and self.gaze is not None
        gaze = {"screen_x": None, "screen_y": None, "on_screen": None}
        if on:
            x, y = np.clip(self.gaze / (self.w, self.h), 0.0, 1.0)
            rx, ry = self.raw_gaze / (self.w, self.h)
            gaze = {"screen_x": round(float(x), 4), "screen_y": round(float(y), 4),
                    "on_screen": bool(-0.05 <= rx <= 1.05 and -0.05 <= ry <= 1.05)}
        return {
            "schema_version": 2,
            "source": "screen_gaze",
            "timestamp": self.timestamp,
            "time": round(t, 2),
            "face": self.face is not None,
            "calibrated": self.model is not None,
            "calibration_error_px": None if self.model is None else round(self.report["error"], 1),
            "head_moved": self.head_moved,
            "screen": [self.w, self.h],
            "gaze": gaze,
        }

    # --- drawing -------------------------------------------------------------------------

    def render(self, t: float) -> np.ndarray:
        img = np.full((self.h, self.w, 3), BACKGROUND, np.uint8)
        {"intro": self._draw_intro, "calibrating": self._draw_calibration,
         "result": self._draw_result, "tracking": self._draw_tracking}[self.mode](img, t)
        return img

    def _draw_intro(self, img: np.ndarray, t: float) -> None:
        s, n = self.s, self.args.points
        seconds = n * (SETTLE_SECONDS + COLLECT_SECONDS) + 1
        put_centered(img, "Screen gaze", int(0.12 * self.h), 1.5 * s, WHITE, 2)
        lines = ("Sit the way you normally do, about an arm's length from the screen.",
                 "Keep your head still and look at each red dot until it disappears.",
                 f"{n} dots, about {seconds:.0f} seconds. Blinking is fine.")
        for i, line in enumerate(lines):
            put_centered(img, line, int((0.19 + 0.045 * i) * self.h), 0.75 * s, GRAY, 1)
        width = int(0.34 * self.w)
        bottom = self._draw_camera(img, (self.w - width) // 2, int(0.34 * self.h), width)
        seen = self.face is not None and measure(self.face.points) is not None
        status, color = ("Eyes found", GREEN) if seen else ("Looking for your face...", YELLOW)
        put_centered(img, status, bottom + int(0.05 * self.h), 0.8 * s, color, 2)
        put_centered(img, "Press SPACE to start      q to quit", int(0.94 * self.h), 0.8 * s, WHITE, 1)

    def _draw_calibration(self, img: np.ndarray, t: float) -> None:
        cal = self.calibration
        if cal.done:
            return
        tx, ty = cal.targets[cal.current]
        centre = (int(tx * self.w), int(ty * self.h))
        settled, recorded = cal.progress(t)
        big, small = 0.035 * self.h, 0.009 * self.h
        radius = small + (big - small) * (1 - settled) ** 2  # shrinks onto the spot to look at
        cv2.circle(img, centre, int(radius), RED, -1, cv2.LINE_AA)
        cv2.circle(img, centre, max(2, int(0.3 * small)), WHITE, -1, cv2.LINE_AA)
        if recorded > 0:
            ring = int(2.2 * small)
            cv2.ellipse(img, centre, (ring, ring), -90, 0, 360 * recorded, WHITE, 2, cv2.LINE_AA)
        put_centered(img, f"{min(cal.shown + 1, cal.total)} / {cal.total}", int(0.975 * self.h), 0.55 * self.s, DARK)
        if self.face is None:
            put_centered(img, "Can't see your face", int(0.5 * self.h + 3 * big), 0.8 * self.s, YELLOW, 2)

    def _draw_result(self, img: np.ndarray, t: float) -> None:
        s, r = self.s, self.report
        if not r["ok"]:
            put_centered(img, "Calibration didn't work", int(0.4 * self.h), 1.2 * s, YELLOW, 2)
            put_centered(img, f"The camera saw your eyes clearly on only {r['usable']} of {r['total']} dots.",
                         int(0.47 * self.h), 0.75 * s, GRAY)
            put_centered(img, "Face the screen with light on your face (no window behind you), then try again.",
                         int(0.52 * self.h), 0.75 * s, GRAY)
            put_centered(img, "c: calibrate again      q: quit", int(0.94 * self.h), 0.8 * s, WHITE)
            return
        for target, mean, _ in r["dots"]:
            a, b = tuple(int(v) for v in target), tuple(int(v) for v in mean)
            cv2.line(img, a, b, DARK, 1, cv2.LINE_AA)
            cv2.circle(img, a, int(0.008 * self.h), RED, -1, cv2.LINE_AA)
            cv2.circle(img, b, int(0.005 * self.h), WHITE, -1, cv2.LINE_AA)
        share = r["error"] / self.w
        verdict = ("good for webcam tracking" if share < 0.06 else "okay" if share < 0.12
                   else "rough: better light and a still head help")
        # The text sits between the rows of dots.
        put_centered(img, "Calibration done", int(0.375 * self.h), 1.2 * s, WHITE, 2)
        put_centered(img, f"Accuracy about {r['error']:.0f} px ({100 * share:.0f}% of the screen width): {verdict}",
                     int(0.42 * self.h), 0.75 * s, GRAY)
        put_centered(img, "Red: where the dots were.  White: where the tracker thinks you looked.",
                     int(0.455 * self.h), 0.65 * s, GRAY)
        put_centered(img, "SPACE: start tracking      c: calibrate again      q: quit", int(0.985 * self.h), 0.8 * s, WHITE)

    def _draw_tracking(self, img: np.ndarray, t: float) -> None:
        s = self.s
        if self.show_grid:
            cell = None if self.gaze is None else (min(int(3 * self.gaze[0] / self.w), 2),
                                                   min(int(3 * self.gaze[1] / self.h), 2))
            for col in range(3):
                for row in range(3):
                    x0, y0 = col * self.w // 3, row * self.h // 3
                    x1, y1 = (col + 1) * self.w // 3, (row + 1) * self.h // 3
                    if cell == (col, row):
                        cv2.rectangle(img, (x0, y0), (x1, y1), HIGHLIGHT, -1)
                    cv2.rectangle(img, (x0, y0), (x1, y1), DARK, 1)
        if self.gaze is not None:
            centre = (int(np.clip(self.gaze[0], 0, self.w - 1)), int(np.clip(self.gaze[1], 0, self.h - 1)))
            cv2.circle(img, centre, int(self.report["error"]), GRAY, 1, cv2.LINE_AA)  # typical error
            cv2.circle(img, centre, int(0.014 * self.h), GREEN, -1, cv2.LINE_AA)
            cv2.circle(img, centre, int(0.014 * self.h), WHITE, 2, cv2.LINE_AA)
        put(img, "c: calibrate again   g: grid   p: camera   q: quit", (int(0.015 * self.w), int(0.035 * self.h)),
            0.6 * s, GRAY)
        warning = ("Can't see your face" if self.face is None else
                   "You moved since calibrating: press c to calibrate again" if self.head_moved else None)
        if warning:
            put_centered(img, warning, int(0.08 * self.h), 0.8 * s, YELLOW, 2)
        if self.show_camera:
            width = int(0.18 * self.w)
            self._draw_camera(img, self.w - width - int(0.015 * self.w), int(0.015 * self.h), width)

    def _draw_camera(self, img: np.ndarray, x: int, y: int, width: int) -> int:
        """Mirrored webcam view with the tracked eye points; returns the bottom edge."""
        if self.frame is None:
            return y
        fh, fw = self.frame.shape[:2]
        height = min(int(width * fh / fw), self.h - y)
        view = cv2.resize(cv2.flip(self.frame, 1), (width, int(width * fh / fw)))[:height]
        if self.face is not None:
            scale = width / fw
            mirror = lambda p: (int((fw - p[0]) * scale), int(p[1] * scale))  # noqa: E731
            for eye in EYES:  # eye corners in yellow, iris centre in green
                cv2.circle(view, mirror(self.face.points[eye[0]]), 2, YELLOW, -1, cv2.LINE_AA)
                cv2.circle(view, mirror(self.face.points[eye[3]]), 2, YELLOW, -1, cv2.LINE_AA)
                cv2.circle(view, mirror(self.face.points[eye[6]]), 3, GREEN, -1, cv2.LINE_AA)
        img[y:y + height, x:x + width] = view
        cv2.rectangle(img, (x - 1, y - 1), (x + width, y + height), DARK, 1)
        return y + height


def put(img, text, org, scale, color=WHITE, thickness=1) -> None:
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def put_centered(img, text, y, scale, color=WHITE, thickness=1) -> None:
    (width, _), _ = cv2.getTextSize(text, FONT, scale, thickness)
    put(img, text, ((img.shape[1] - width) // 2, y), scale, color, thickness)


def screen_size() -> tuple[int, int]:
    """Size of the main screen in (logical) pixels, so the full-screen image isn't distorted."""
    try:
        if sys.platform == "darwin":
            cg = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
            cg.CGMainDisplayID.restype = ctypes.c_uint32
            for f in (cg.CGDisplayPixelsWide, cg.CGDisplayPixelsHigh):
                f.argtypes, f.restype = [ctypes.c_uint32], ctypes.c_size_t
            display = cg.CGMainDisplayID()
            return int(cg.CGDisplayPixelsWide(display)), int(cg.CGDisplayPixelsHigh(display))
        if sys.platform == "win32":
            user32 = ctypes.windll.user32
            return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except (OSError, AttributeError):
        pass
    return 1280, 720


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=int(os.environ.get("STRESS_CAMERA", 0)),
                        help="webcam number (default 0, or the STRESS_CAMERA environment variable)")
    parser.add_argument("--points", type=int, choices=(9, 13, 16), default=13,
                        help="number of calibration dots (default 13)")
    parser.add_argument("--smoothing", type=float, default=1.0,
                        help="lower = steadier but slower gaze dot (default 1.0)")
    parser.add_argument("--udp", default=f"127.0.0.1:{GAZE_PORT}", help="host:port to send gaze to, or 'off'")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    size = screen_size()
    detector = FaceDetector(ensure_model(FACE_MODEL))
    cap = open_camera(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # a sharper eye region than 640x480
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    tracker = ScreenGazeTracker(args, size)
    if tracker.udp:
        print(f"Sending gaze to udp://{args.udp}")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    start, failures = time.monotonic(), 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                failures += 1
                if failures > 50:
                    raise SystemExit("The webcam stopped sending frames.")
                time.sleep(0.02)
                continue
            failures = 0
            timestamp, t = time.time(), time.monotonic() - start
            tracker.step(t, frame, detector.detect(frame, t), timestamp)
            cv2.imshow(WINDOW, tracker.render(t))
            k = cv2.waitKey(1) & 0xFF
            if k != 255 and not tracker.key(k, t):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        detector.close()
        if tracker.udp:
            tracker.udp.close()
        cv2.destroyAllWindows()
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
