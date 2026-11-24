import argparse
import math
import unittest

import numpy as np

import screen_gaze as sg
from eye_tracking import EYES
from face_features import Face

SIZE = (1470, 956)


def fake_face(gx, gy, rng, blink=False, noise=0.6, yaw=0.0):
    """A face whose irises (and gaze blendshapes) follow a known screen point gx, gy in 0..1."""
    points = np.zeros((478, 2)) + (600, 400)
    points[234], points[454] = (480, 400), (760, 400)  # face edges: face width 280 px
    dx = 14 * math.tanh(1.6 * (gx - 0.5))              # nonlinear, like a real eye
    dy = 5 * (gy - 0.5) + 1.5 * (gx - 0.5) ** 2
    lid = 1.0 if blink else 8 - 3 * (gy - 0.5)          # lids drop when looking down
    for left, indices in zip((520, 660), EYES):
        corner_l, up1, up2, corner_r, low2, low1, iris = indices
        points[corner_l], points[corner_r] = (left, 400), (left + 60, 400)
        points[up1], points[up2] = (left + 20, 400 - lid), (left + 40, 400 - lid)
        points[low1], points[low2] = (left + 20, 400 + lid), (left + 40, 400 + lid)
        points[iris] = np.array((left + 30 + dx, 400 + dy)) + rng.normal(0, noise, 2)
    shapes = {"eyeLookOutLeft": max(0, gx - .5), "eyeLookInLeft": max(0, .5 - gx),
              "eyeLookInRight": max(0, gx - .5), "eyeLookOutRight": max(0, .5 - gx),
              "eyeLookUpLeft": max(0, .5 - gy), "eyeLookUpRight": max(0, .5 - gy),
              "eyeLookDownLeft": max(0, gy - .5), "eyeLookDownRight": max(0, gy - .5)}
    shapes = {name: value + rng.normal(0, 0.02) for name, value in shapes.items()}
    return Face(points.astype(np.float32), shapes, (0.0, yaw, 0.0))


def new_tracker():
    return sg.ScreenGazeTracker(argparse.Namespace(points=13, smoothing=1.0, udp="off"), SIZE)


def calibrate(tracker, rng, t=0.0, face=fake_face):
    """Runs a whole calibration at 30 fps, looking at each dot once it has shrunk a little."""
    frame = np.zeros((720, 1280, 3), np.uint8)
    tracker.key(ord(" "), t)
    while tracker.mode == "calibrating":
        t += 1 / 30
        cal = tracker.calibration
        settled, _ = cal.progress(t)
        gx, gy = cal.targets[cal.current] if settled >= 0.5 else (0.5, 0.5)
        tracker.step(t, frame, face(gx, gy, rng, blink=rng.random() < 0.05) if face else None)
    return t


class ScreenGazeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.default_rng(7)
        cls.tracker = new_tracker()
        cls.t = calibrate(cls.tracker, cls.rng)
        cls.frame = np.zeros((720, 1280, 3), np.uint8)

    def look(self, gx, gy, frames=45, **kwargs):
        for _ in range(frames):
            self.__class__.t += 1 / 30
            self.tracker.step(self.t, self.frame, fake_face(gx, gy, self.rng, **kwargs))

    def test_features_follow_the_eyes(self):
        rng = np.random.default_rng(0)
        left, right = (sg.eye_features(fake_face(x, 0.5, rng, noise=0)) for x in (0.2, 0.8))
        down = sg.eye_features(fake_face(0.5, 0.9, rng, noise=0))
        self.assertGreater(right[0], left[0])
        self.assertLess(down[2], left[2])

    def test_calibration_learns_the_mapping(self):
        report = self.tracker.report
        self.assertTrue(report["ok"])
        self.assertEqual(report["usable"], 13)
        self.assertLess(report["error"], 60)  # px, measured on left-out dots

    def test_tracks_points_between_the_dots(self):
        self.tracker.key(ord(" "), self.t)
        self.assertEqual(self.tracker.mode, "tracking")
        for gx, gy in ((0.3, 0.6), (0.62, 0.22), (0.85, 0.8)):
            self.look(gx, gy)
            self.assertLess(np.linalg.norm(self.tracker.gaze - (gx * SIZE[0], gy * SIZE[1])), 60)

    def test_blink_keeps_the_last_point(self):
        self.tracker.key(ord(" "), self.t)
        self.look(0.4, 0.4)
        before = self.tracker.gaze.copy()
        self.look(0.9, 0.9, frames=3, blink=True)
        np.testing.assert_allclose(self.tracker.gaze, before)

    def test_head_turn_is_flagged(self):
        self.tracker.key(ord(" "), self.t)
        self.look(0.5, 0.5, frames=3)
        self.assertFalse(self.tracker.head_moved)
        self.look(0.5, 0.5, frames=3, yaw=15.0)
        self.assertTrue(self.tracker.head_moved)
        self.look(0.5, 0.5, frames=3)

    def test_message(self):
        self.tracker.key(ord(" "), self.t)
        self.look(0.25, 0.75)
        message = self.tracker.message(self.t)
        self.assertEqual(message["schema_version"], 2)
        self.assertEqual(message["source"], "screen_gaze")
        self.assertTrue(message["calibrated"] and message["gaze"]["on_screen"])
        self.assertAlmostEqual(message["gaze"]["screen_x"], 0.25, delta=0.05)
        self.assertAlmostEqual(message["gaze"]["screen_y"], 0.75, delta=0.05)

    def test_every_screen_renders(self):
        for mode in ("result", "tracking"):
            self.tracker.mode = mode
            self.assertEqual(self.tracker.render(self.t).shape, (SIZE[1], SIZE[0], 3))
        tracker = new_tracker()
        tracker.step(0.0, self.frame, fake_face(0.5, 0.5, self.rng))
        self.assertEqual(tracker.render(0.0).shape, (SIZE[1], SIZE[0], 3))
        tracker.key(ord(" "), 0.0)
        tracker.step(0.1, self.frame, None)
        self.assertEqual(tracker.render(0.1).shape, (SIZE[1], SIZE[0], 3))


class NoFaceTest(unittest.TestCase):
    def test_calibration_stops_without_a_face(self):
        tracker = new_tracker()
        t = calibrate(tracker, np.random.default_rng(1), face=None)
        self.assertLess(t, sg.NO_FACE_SECONDS + 1)
        self.assertFalse(tracker.report["ok"])
        self.assertIsNone(tracker.message(t)["gaze"]["screen_x"])


if __name__ == "__main__":
    unittest.main()
