"""Webcam stress detector for games (stress-test-gaming-team).

  python stress_cam.py                  detect stress live (calibrates first)
  python stress_cam.py record calm      record a labeled session for training
  python stress_cam.py record stressed
  python stress_cam.py train            train your personal model on the recordings

While running, the stress level is sent as JSON over UDP (127.0.0.1:5005 by
default) so a game in any engine can react to it; see game_receiver_example.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import socket
import sys
import time
from collections import deque
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pylsl

from eye_tracking import EyeTracker
from face_features import FaceDetector, FeatureExtractor, ensure_model
from pulse import PulseEstimator
from stress_model import (LABELS, READABLE, BaselineScorer, EyeScorer, FocusScorer, PersonalModel, Recorder,
                           Smoother, focus_level, level, train)

HERE = Path(__file__).resolve().parent
FACE_MODEL = HERE / "models" / "face_landmarker.task"
PERSONAL_MODEL = HERE / "models" / "personal_model.joblib"
RECORDINGS = HERE / "recordings"

WINDOW = "Stress detector"
RECORD_WARMUP = 10.0  # seconds before a recording starts, so blink rate and pulse have data
HISTORY_SECONDS = 60
LABEL_KEYS = {ord("0"): "none", ord("1"): "calm", ord("2"): "stressed", ord("3"): "focused", ord("4"): "distracted"}
OFF_SCREEN_Z = 2.5                # a gaze z-score beyond this (vs. your calibration) counts as "looking away"
OFF_SCREEN_WINDOW_SECONDS = 30.0  # window for the off-screen-time cue

# OpenCV colors are (blue, green, red).
GREEN, YELLOW, RED = (90, 200, 90), (0, 200, 255), (60, 60, 235)
WHITE, GRAY = (255, 255, 255), (160, 160, 160)
LEVEL_COLORS = {"calm": GREEN, "tense": YELLOW, "stressed": RED}


class StressApp:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.detector = FaceDetector(ensure_model(FACE_MODEL))
        self.pulse = None if args.no_pulse else PulseEstimator()
        self.extractor = FeatureExtractor(pulse=self.pulse)
        self.recorder = self.scorer = self.udp = self.label_recorder = self.lsl_outlet = None
        if args.command == "record":
            self.recorder = Recorder(RECORDINGS, args.label)
            print(f"Recording '{args.label}' for {args.seconds:.0f} s after a {RECORD_WARMUP:.0f} s warm-up.")
        else:
            if PERSONAL_MODEL.exists() and not args.baseline:
                self.scorer = PersonalModel(PERSONAL_MODEL)
                print("Using your personal model (add --baseline to calibrate instead).")
            else:
                self.scorer = BaselineScorer(args.calibration, args.sensitivity)
                print(f"Calibrating for {args.calibration:.0f} s: relax and look at the screen like you normally do.")
            self.smoother = Smoother(args.smoothing)
            self.focus_scorer = FocusScorer(args.calibration)
            self.focus_smoother = Smoother(args.smoothing)
            self.eye_scorer = EyeScorer(args.calibration)
            self._eye_seen, self._off_screen, self._last_eye_t = deque(), deque(), None
            if args.udp != "off":
                try:
                    self.udp = UdpSender(args.udp)
                except (OSError, ValueError, OverflowError) as exc:
                    print(f"UDP disabled: {exc}", file=sys.stderr)
                else:
                    print(f"Sending stress values to udp://{args.udp}")

        self.eye_tracker = None if self.recorder else EyeTracker(args.eye_calibration, args.attention_window)
        self.eye_data = {}
        self.timestamp = None
        self.timings = deque(maxlen=600)
        self.capture_failures = 0
        self.t, self.face, self.feats = 0.0, None, None
        self.stress, self.reasons = None, {}
        self.focus = None
        self.label = "none"
        self.history = deque()
        self.fps, self.details = 0.0, False
        self.test_cues = getattr(args, "test_cues", False)

    def run(self, on_step: Callable[[], None] | None = None) -> None:
        """Runs until the window is closed / max-seconds is reached / recording finishes.

        on_step, if given, runs right after every step() (e.g. server.py uses it to drain
        its command queue and publish the latest metrics, from the same thread as step()).
        """
        if not self.args.no_window:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        last_frame, last_print, last_cue_print = None, -math.inf, -math.inf
        try:
            for t, frame in frames(self.args):
                self.step(t, frame)
                if frame is None:
                    self.capture_failures += 1
                    if self.done():
                        break
                    continue
                now = time.perf_counter()
                if last_frame is not None:
                    rate = 1.0 / max(now - last_frame, 1e-6)
                    self.fps = 0.9 * self.fps + 0.1 * rate if self.fps else rate
                last_frame = now
                if self.test_cues and not self.recorder and t - last_cue_print >= 0.2:
                    last_cue_print = t
                    print(self.cue_debug_line())
                if self.args.no_window:
                    if t - last_print >= 1.0:
                        last_print = t
                        print(self.status())
                elif frame is not None and not self.show(frame):
                    break
                if self.done():
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.detector.close()
            if self.udp:
                self.udp.close()
            if self.args.command == 'run' and self.args.profile:
                self.print_profile()
            cv2.destroyAllWindows()
            if self.recorder:
                self.recorder.close()
                if self.recorder.rows:
                    print(f"Saved {self.recorder.rows} samples to {self.recorder.path}")
                else:
                    print("Nothing was recorded (no face seen after the warm-up).")
            if self.label_recorder:
                self.label_recorder.close()
                if self.label_recorder.rows:
                    print(f"Saved {self.label_recorder.rows} rows to {self.label_recorder.path}")
                else:
                    print("Nothing was recorded (no frames were captured).")

    def step(self, t: float, frame: np.ndarray | None) -> None:
        self.t = t
        self.timestamp = time.time()
        start = time.perf_counter()
        failure = frame is None
        self.face = None
        if frame is not None:
            try:
                self.face = self.detector.detect(frame, t)
            except (RuntimeError, ValueError, cv2.error):
                failure = True
        if self.face is not None and not np.isfinite(self.face.points).all():
            self.face, failure = None, True
        detected = time.perf_counter()
        if self.eye_tracker:
            self.eye_data = self.eye_tracker.update(
                t, self.face.points if self.face else None,
                self.face.pose if self.face else None, failure,
            )
        eyes_done = time.perf_counter()
        if frame is None:
            self.extractor.update(t, frame, None)
            self.feats = None
            if self.udp:
                self.udp.send(t, self.message())
            return
        self.extractor.update(t, frame, self.face)
        self.feats = self.extractor.features(t)

        if self.recorder:
            if self.feats and t >= RECORD_WARMUP:
                self.recorder.write(t, self.feats)
            return
        if self.feats:
            if self.scorer.ready:
                value, self.reasons = self.scorer.score(self.feats)
                self.stress = self.smoother.update(t, value)
            else:
                self.scorer.calibrate(t, self.feats)
            if self.focus_scorer.ready:
                fvalue, _ = self.focus_scorer.score(self.feats)
                self.focus = self.focus_smoother.update(t, fvalue)
            else:
                self.focus_scorer.calibrate(t, self.feats)
            if self.face is not None:  # only on frames with a real (not window-carried) measurement
                if self.eye_scorer.ready:
                    self._update_eye_tracking(t)
                else:
                    self.eye_scorer.calibrate(t, self.feats)
        if self.stress is not None:
            self.history.append((t, self.stress))
            while t - self.history[0][0] > HISTORY_SECONDS:
                self.history.popleft()
        scored = time.perf_counter()
        if self.udp and self.udp.due(t):
            self.udp.send(t, self.message())
        if self.args.profile:
            finished = time.perf_counter()
            self.timings.append((t, detected - start, eyes_done - detected,
                                 scored - eyes_done, finished - scored, finished - start))

    def print_profile(self) -> None:
        print(f"Failed camera reads: {self.capture_failures}")
        if not self.timings:
            return
        values = np.asarray(self.timings)
        if len(values) > 1:
            span = values[-1, 0] - values[0, 0]
            if span:
                print(f"Frames per source second: {(len(values) - 1) / span:.1f}; loop FPS: {self.fps:.1f}")
        for name, column in zip(("landmarker", "eyes", "legacy", "publish", "step"), values[:, 1:].T):
            p50, p95 = np.percentile(column * 1000, (50, 95))
            print(f"{name:12s} p50 {p50:6.2f} ms  p95 {p95:6.2f} ms")

    def done(self) -> bool:
        if self.recorder and self.t >= RECORD_WARMUP + self.args.seconds:
            return True
        return self.args.max_seconds is not None and self.t >= self.args.max_seconds

    def message(self) -> dict:
        """What the game receives, up to 20 times per second."""
        def number(name):
            value = self.feats[name] if self.feats else math.nan
            return None if not math.isfinite(value) else round(value, 1)

        stress = self.stress if self.stress is not None and math.isfinite(self.stress) else None
        return {
            "stress": None if stress is None else round(stress, 3),
            "level": None if stress is None else level(stress),
            "face": self.face is not None,
            "calibrating": not self.scorer.ready,
            "focus_calibrating": not self.focus_scorer.ready,
            "heart_rate": number("heart_rate"),
            "blink_rate": number("blink_rate"),
            **{name: number(name) for name in CUE_NAMES},
            "time": round(self.t, 2),
            "schema_version": 2,
            "timestamp": self.timestamp,
            "fps": round(self.fps, 1),
            "stress_analysis": {
                "score": None, "confidence": None, "method": "unvalidated",
                "legacy_method": self.scorer.name,
                "legacy_contributions": {
                    name: value if math.isfinite(value) else None
                    for name, value in self.reasons.items() if self.face is not None and self.feats
                },
                "behavioral_features": {
                    name: value if math.isfinite(value) else None
                    for name, value in (self.feats or {}).items() if self.face is not None
                },
            },
            **self.eye_data,
        }

    def top_reasons(self, n: int = 2) -> list[str]:
        ranked = sorted(self.reasons.items(), key=lambda kv: -kv[1])
        return [READABLE[name] for name, push in ranked[:n] if push > 0.25]

    def cue_debug_line(self) -> str:
        """One console line for --test-cues: raw blendshape inputs and computed cue values."""
        head = f"[{self.t:6.1f}s]" + ("" if self.face else " (no face)")
        if not self.face:
            return head
        dbg = self.extractor.cues.debug
        inputs = " ".join(f"{k}={v:.2f}" for k, v in dbg["inputs"].items())
        cues = " ".join(f"{name} raw={dbg['raw'][name]:.2f} on={dbg['gated'][name]:.2f}" for name in CUE_NAMES)
        return f"{head} {inputs} | {cues}" + (" SPEAKING/CHEWING" if dbg["speaking"] else "")

    def status(self) -> str:
        """One console line, printed every second with --no-window."""
        head = f"[{self.t:6.1f}s]" + ("" if self.face else " (no face)")
        if self.recorder:
            if self.t < RECORD_WARMUP:
                return f"{head} recording starts in {RECORD_WARMUP - self.t:.0f} s"
            return f"{head} recording '{self.recorder.label}': {self.recorder.rows} samples"
        if not self.scorer.ready:
            return f"{head} calibrating {self.scorer.progress(self.t):.0%}"
        if self.stress is None:
            return f"{head} waiting for a face"
        m = {k: "--" if v is None else v for k, v in self.message().items()}
        label = f" label {self.label}" if self.label_recorder else ""
        return (f"{head} stress {self.stress:.2f} {m['level']:<8} heart {m['heart_rate']} bpm, "
                f"blinks {m['blink_rate']}/min, why: {', '.join(self.top_reasons()) or '-'}{label}")

    # ---- control (also used by server.py, from the same thread as step()) ----

    def set_label(self, label: str, t: float | None = None) -> None:
        if label != self.label:
            print(f"[{(self.t if t is None else t):6.1f}s] label -> {label}")
            self.label = label

    def recalibrate(self) -> None:
        """Restarts calibration for every baseline-based scorer. No-op with a personal model."""
        if not isinstance(self.scorer, BaselineScorer):
            return
        self.scorer.restart()
        self.smoother.reset()
        self.stress, self.reasons = None, {}
        self.history.clear()
        self.focus_scorer.restart()
        self.focus_smoother.reset()
        self.focus = None
        self.eye_scorer.restart()
        self._eye_seen.clear()
        self._off_screen.clear()
        self._last_eye_t = None

    def start_recording(self, name: str) -> None:
        if self.label_recorder:
            self.label_recorder.close()
        self.label_recorder = LabelRecorder(RECORDINGS, name)

    def stop_recording(self) -> dict:
        if not self.label_recorder:
            return {"rows": 0, "path": None}
        path, rows = self.label_recorder.path, self.label_recorder.rows
        self.label_recorder.close()
        self.label_recorder = None
        return {"rows": rows, "path": None if rows == 0 else str(path)}

    def _update_eye_tracking(self, t: float) -> None:
        """Accumulates off-screen time: a gaze z-score (vs. your calibration) beyond
        OFF_SCREEN_Z, over the last OFF_SCREEN_WINDOW_SECONDS.
        """
        dev = self.eye_scorer.deviations(self.feats)
        off_screen = not math.isnan(dev["gaze_x"]) and (abs(dev["gaze_x"]) > OFF_SCREEN_Z
                                                          or abs(dev["gaze_y"]) > OFF_SCREEN_Z)
        if self._last_eye_t is not None and 0 < t - self._last_eye_t < 0.5:
            dt = t - self._last_eye_t
            self._eye_seen.append((t, dt))
            if off_screen:
                self._off_screen.append((t, dt))
        self._last_eye_t = t
        for q in (self._eye_seen, self._off_screen):
            while q and t - q[0][0] > OFF_SCREEN_WINDOW_SECONDS:
                q.popleft()

    def off_screen_seconds(self) -> float:
        return sum(dt for _, dt in self._off_screen)

    def eye_cues(self) -> dict:
        """Eye/gaze cues for the server: each relative to your calibration baseline,
        except long_blink_count and off_screen_seconds, which are raw counts/durations
        (off-screen detection itself is baseline-relative; see _update_eye_tracking).
        """
        if not self.eye_scorer.ready or self.feats is None:
            return {"ready": False, "gaze_x": None, "gaze_y": None, "eye_wide": None,
                    "blink_rate": None, "long_blink_count": None, "off_screen_seconds": None}

        def num(x):
            return None if x is None or math.isnan(x) else round(float(x), 3)

        dev = self.eye_scorer.deviations(self.feats)
        return {
            "ready": True,
            "gaze_x": num(dev["gaze_x"]),
            "gaze_y": num(dev["gaze_y"]),
            "eye_wide": num(dev["eye_wide"]),
            "blink_rate": num(dev["blink_rate"]),
            "long_blink_count": self.extractor.long_blink_count(self.t),
            "off_screen_seconds": round(self.off_screen_seconds(), 1),
        }

    # ---- window ---------------------------------------------------------------

    def show(self, frame: np.ndarray) -> bool:
        """Draws the overlay and handles keys. Returns False when the user quits."""
        self.draw(frame)
        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return False
        if key == ord("d"):
            self.details = not self.details
        if key == ord("c") and self.eye_tracker:
            self.eye_tracker.reset()
        if key == ord("c") and isinstance(self.scorer, BaselineScorer):
            self.recalibrate()
        if key in LABEL_KEYS:
            self.set_label(LABEL_KEYS[key])
        return True

    def draw(self, img: np.ndarray) -> None:
        h, w = img.shape[:2]
        color = LEVEL_COLORS[level(self.stress)] if self.stress is not None else GRAY
        if self.face is not None:
            x0, y0, x1, y1 = self.face.box()
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
            if self.details and self.pulse is not None:
                for bx0, by0, bx1, by1 in self.pulse.rois:
                    cv2.rectangle(img, (bx0, by0), (bx1, by1), (255, 0, 255), 1)
        else:
            put(img, "No face - look at the camera", (w // 2 - 160, h // 2), 0.7, YELLOW, 2)

        shade(img, 0, 0, 270, 136 + (24 if self.label_recorder else 0))
        if self.recorder:
            self._draw_recording(img)
        elif not self.scorer.ready:
            put(img, "Calibrating...", (12, 32), 0.8, YELLOW, 2)
            put(img, "relax, look at the screen", (12, 58))
            bar(img, 12, 70, 246, self.scorer.progress(self.t), YELLOW)
        elif self.stress is not None:
            put(img, f"LEGACY {self.stress:4.0%}", (12, 32), 0.8, color, 2)
            bar(img, 12, 44, 246, self.stress, color)
            put(img, level(self.stress).upper(), (12, 82), 0.6, color, 2)
            put(img, ", ".join(self.top_reasons()), (12, 104), 0.45)

        hr = self.feats["heart_rate"] if self.feats else math.nan
        blinks = self.feats["blink_rate"] if self.feats else math.nan
        put(img, f"heart {'--' if math.isnan(hr) else f'{hr:.0f}'} bpm (beta)   "
                 f"blinks {'--' if math.isnan(blinks) else f'{blinks:.0f}'}/min", (12, 126), 0.42, GRAY)
        if self.label_recorder:
            put(img, f"LABEL: {self.label.upper()}", (12, 150), 0.55, YELLOW, 2)

        mode = "recording" if self.recorder else self.scorer.name
        put_right(img, f"{mode} | {self.fps:.0f} fps", 20)
        if self.eye_data:
            self._draw_eyes(img, 52)
        keys = "q quit  d details" + ("  c recalibrate" if self.eye_tracker else "")
        put_right(img, keys, 40)
        if self.details:
            self._draw_details(img)
        if not self.recorder:
            self._draw_history(img)

    def _draw_eyes(self, img: np.ndarray, top: int) -> None:
        data = self.eye_data
        gaze = data["gaze"]
        progress = data["eye_calibration_progress"]
        right = img.shape[1] - (208 if self.details else 10)
        left = max(0, right - 156)
        shade(img, left, top, right, min(img.shape[0], top + 144), 0.7)
        if progress < 1:
            put(img, f"Look center {progress:.0%}", (left + 6, top + 19), 0.42)
            return

        direction = gaze["direction"] or data["tracking_state"].replace("_", " ")
        put(img, f"Eyes: {direction}", (left + 6, top + 19), 0.43)
        center = (left + 32, top + 53)
        cv2.rectangle(img, (center[0] - 24, center[1] - 24),
                      (center[0] + 24, center[1] + 24), GRAY, 1)
        cv2.line(img, (center[0] - 24, center[1]), (center[0] + 24, center[1]), GRAY)
        cv2.line(img, (center[0], center[1] - 24), (center[0], center[1] + 24), GRAY)
        dx, dy = gaze["center_offset_x"], gaze["center_offset_y"]
        if dx is not None and dy is not None:
            x = center[0] + int(np.clip(dx * 180, -23, 23))
            y = center[1] + int(np.clip(dy * 180, -23, 23))
            cv2.circle(img, (x, y), 4, GREEN, -1)
            put(img, f"x {dx:+.3f}", (left + 64, top + 47), 0.39)
            put(img, f"y {dy:+.3f}", (left + 64, top + 65), 0.39)

        pose = data["head_pose"]
        if pose is not None:
            put(img, f"Head yaw {pose['yaw_deg']:+.0f} deg", (left + 6, top + 98), 0.4)
        score = data["attention"]["score"]
        if score is not None:
            put(img, f"Center time {score:.0%}", (left + 6, top + 120), 0.4)

    def _draw_recording(self, img: np.ndarray) -> None:
        label = self.recorder.label
        if self.t < RECORD_WARMUP:
            put(img, f"Get ready: {RECORD_WARMUP - self.t:.0f}", (12, 32), 0.8, YELLOW, 2)
            put(img, f"then recording '{label}'", (12, 58))
        else:
            elapsed = self.t - RECORD_WARMUP
            put(img, f"REC {label}", (12, 32), 0.8, RED, 2)
            bar(img, 12, 44, 246, elapsed / self.args.seconds, RED)
            put(img, f"{elapsed:.0f} / {self.args.seconds:.0f} s   {self.recorder.rows} samples", (12, 82))

    def _draw_details(self, img: np.ndarray) -> None:
        if not self.feats:
            return
        w = img.shape[1]
        x0, y0 = w - 200, 52
        shade(img, x0, y0, w, y0 + 18 * len(self.feats) + 8)
        for i, (name, value) in enumerate(self.feats.items()):
            y = y0 + 18 * (i + 1)
            put(img, name, (x0 + 8, y), 0.42, GRAY)
            put(img, "--" if math.isnan(value) else f"{value:.2f}", (x0 + 140, y), 0.42)

    def _draw_history(self, img: np.ndarray) -> None:
        h, w = img.shape[:2]
        x0, y0, x1, y1 = 10, h - 60, w - 10, h - 10
        shade(img, x0, y0, x1, y1)
        for threshold in (0.35, 0.65):
            y = int(y1 - threshold * (y1 - y0))
            cv2.line(img, (x0, y), (x1, y), (80, 80, 80), 1)
        points = [(int(x1 - (self.t - t) / HISTORY_SECONDS * (x1 - x0)), int(y1 - s * (y1 - y0)))
                  for t, s in self.history]
        if len(points) > 1:
            cv2.polylines(img, [np.array(points, np.int32)], False, WHITE, 1, cv2.LINE_AA)
        put(img, f"stress, last {HISTORY_SECONDS} s", (x0 + 6, y0 + 14), 0.4, GRAY)


class UdpSender:
    """Sends JSON messages to a game over UDP, at most `rate` times per second."""

    def __init__(self, address: str, rate: float = 20.0):
        host, port = address.rsplit(":", 1)
        self.interval, self._last = 1.0 / rate, -math.inf
        self.target = socket.getaddrinfo(host, int(port), socket.AF_INET, socket.SOCK_DGRAM)[0][4]
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setblocking(False)

    def close(self) -> None:
        self._sock.close()

    def due(self, t: float) -> bool:
        return t - self._last >= self.interval

    def send(self, t: float, message: dict) -> None:
        if not self.due(t):
            return
        self._last = t
        message = {**message, "t_emit": round(clock.now(), 6)}  # stamped now, not at frame capture
        try:
            self._sock.sendto(json.dumps(message, allow_nan=False).encode(), self.target)
        except OSError:
            pass  # nobody is listening yet; that's fine


class StressOutlet:
    """Publishes stress and focus as an LSL stream, so LabRecorder can record them
    alongside the Muse 2 EEG (same clock.now() timestamps, see clock.py).
    """

    def __init__(self):
        info = pylsl.StreamInfo(name="WebcamStress", type="Stress", channel_count=2,
                                 nominal_srate=pylsl.IRREGULAR_RATE, channel_format="float32",
                                 source_id="facial-analysis-stress-cam")
        channels = info.desc().append_child("channels")
        for label in ("stress", "focus"):
            ch = channels.append_child("channel")
            ch.append_child_value("label", label)
            ch.append_child_value("unit", "normalized")
            ch.append_child_value("type", "Stress")
        self._outlet = pylsl.StreamOutlet(info)

    def push(self, t_capture: float, stress: float | None, focus: float | None) -> None:
        s = math.nan if stress is None else stress
        f = math.nan if focus is None else focus
        self._outlet.push_sample([s, f], timestamp=t_capture)


class LabelRecorder:
    """Per-frame CSV for syncing with an EEG recording in LabRecorder: one row per frame,
    an LSL timestamp, every stress cue, the raw blendshapes behind them, and a label the
    caller sets live (StressApp.label, via the number keys or the server's /label/{name}).
    A frame with no face gets blank cue and score columns rather than the last known
    values, so a look-away doesn't look like data.
    """

    def __init__(self, folder: Path, name: str):
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / f"{name}_{time.strftime('%Y%m%d-%H%M%S')}.csv"
        self.rows = 0
        self._file = open(self.path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["t_capture", "frame_id", "face_found", "stress", "focus",
                                *CUE_NAMES, *RAW_BLENDSHAPE_NAMES, "label"])

    def write(self, t_capture: float, frame_id: int, face, stress: float | None, focus: float | None,
              feats: dict | None, label: str) -> None:
        face_found = face is not None

        def num(x) -> str:
            return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.5g}"

        if face_found:
            cues = [num(feats[name]) for name in CUE_NAMES] if feats else ["" for _ in CUE_NAMES]
            raw = [num(face.blendshapes.get(name, math.nan)) for name in RAW_BLENDSHAPE_NAMES]
        else:
            cues = ["" for _ in CUE_NAMES]
            raw = ["" for _ in RAW_BLENDSHAPE_NAMES]
        self._writer.writerow([f"{t_capture:.6f}", frame_id, int(face_found), num(stress), num(focus),
                                *cues, *raw, label])
        self.rows += 1

    def close(self) -> None:
        self._file.close()
        if self.rows == 0:
            self.path.unlink()


def frames(args: argparse.Namespace):
    """Yields (seconds since start, frame_id, BGR image, t_capture) from the webcam or a video file.

    t_capture is stamped the instant the frame is grabbed, on the shared LSL clock
    (clock.now()), so it lines up with anything else recorded on that clock (e.g. the Muse 2
    EEG in LabRecorder). `t` stays the pre-existing elapsed-seconds-since-start value that the
    rest of the app (calibration, smoothing, history) depends on.
    """
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit(f"Could not open video file {args.video}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    else:
        cap = open_camera(args.camera)
    start, index, failures = clock.now(), 0, 0
    try:
        while True:
            ok, frame = cap.read()
            t_capture = clock.now()
            if not ok:
                if args.video:
                    return
                failures += 1
                if failures > 50:
                    raise SystemExit("The webcam stopped sending frames.")
                yield time.monotonic() - start, None
                time.sleep(0.02)
                continue
            failures = 0
            if args.video:
                yield index / fps, index, frame, t_capture  # video time, so results don't depend on processing speed
            else:
                yield t_capture - start, index, cv2.flip(frame, 1), t_capture  # mirrored, like a selfie
            index += 1
    finally:
        cap.release()


def open_camera(index: int) -> cv2.VideoCapture:
    # DirectShow opens much faster than the default backend on many Windows laptops.
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return cap
        cap.release()
    raise SystemExit(f"Could not open webcam {index}. Try --camera 1, and close other apps using the camera.")


def put(img, text, org, scale=0.5, color=WHITE, thickness=1) -> None:
    """Text with a dark drop shadow so it is readable on any background."""
    x, y = org
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def put_right(img, text, y, scale=0.45, color=GRAY) -> None:
    width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
    put(img, text, (img.shape[1] - width - 10, y), scale, color)


def shade(img, x0, y0, x1, y1, darkness=0.55) -> None:
    region = img[y0:y1, x0:x1]
    region[:] = (region * (1 - darkness)).astype(np.uint8)


def bar(img, x, y, width, fraction, color, height=12) -> None:
    fraction = min(max(fraction, 0.0), 1.0)
    cv2.rectangle(img, (x, y), (x + width, y + height), GRAY, 1)
    cv2.rectangle(img, (x, y), (x + int(width * fraction), y + height), color, -1)


def parse_args(argv: list[str]) -> argparse.Namespace:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--camera", type=int, default=0, help="webcam number (default 0)")
    common.add_argument("--video", help="read a video file instead of the webcam")
    common.add_argument("--no-window", action="store_true", help="no camera window; print to the console")
    common.add_argument("--no-pulse", action="store_true", help="turn off the experimental heart-rate estimate")
    common.add_argument("--max-seconds", type=float, help="stop after this many seconds")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command")
    live = commands.add_parser("run", parents=[common], help="detect stress live (the default)")
    live.add_argument("--baseline", action="store_true", help="ignore your personal model and calibrate instead")
    live.add_argument("--calibration", type=float, default=20.0, help="calibration length in seconds (default 20)")
    live.add_argument("--sensitivity", type=float, default=0.4, help="how strongly the baseline score reacts (default 0.4)")
    live.add_argument("--smoothing", type=float, default=1.0, help="seconds of smoothing on the output (default 1)")
    live.add_argument("--udp", default="127.0.0.1:5005", help="host:port to send stress values to, or 'off'")
    live.add_argument("--eye-calibration", type=float, default=3.0)
    live.add_argument("--attention-window", type=float, default=3.0)
    live.add_argument("--profile", action="store_true", help="print processing latency at shutdown")
    record = commands.add_parser("record", parents=[common], help="record a labeled session for training")
    record.add_argument("label", choices=LABELS)
    record.add_argument("--seconds", type=float, default=120.0, help="length of the recording (default 120)")
    commands.add_parser("train", help="train your personal model on everything in recordings/")

    if not argv or argv[0] not in ("run", "record", "train", "-h", "--help"):
        argv = ["run", *argv]
    args = parser.parse_args(argv)
    if args.command == "run":
        for name in ("eye_calibration", "attention_window", "calibration"):
            if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
                parser.error(f"--{name.replace('_', '-')} must be finite and positive")
        for name in ("sensitivity", "smoothing"):
            if not math.isfinite(getattr(args, name)):
                parser.error(f"--{name} must be finite")
    return args


def main() -> None:
    args = parse_args(sys.argv[1:])
    if args.command == "train":
        train(RECORDINGS, PERSONAL_MODEL)
    else:
        StressApp(args).run()


if __name__ == "__main__":
    main()
