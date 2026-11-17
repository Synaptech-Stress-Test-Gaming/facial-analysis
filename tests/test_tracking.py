import json
import socket
import unittest
from unittest.mock import patch

import numpy as np

from eye_tracking import EYES, EyeTracker, measure
from face_features import Face
from stress_cam import StressApp, UdpSender, parse_args

POSE = (0.0, 0.0, 0.0)


def landmarks(dx=0, lid=3):
    points = np.zeros((478, 2))
    for offset, indices in zip((20, 60), EYES):
        points[list(indices)] = np.array([
            (0, 0), (6, -lid), (14, -lid), (20, 0),
            (14, lid), (6, lid), (10 + dx, 0),
        ]) + (offset, 40)
    return points


class EyeTests(unittest.TestCase):
    def setUp(self):
        self.tracker = EyeTracker(calibration=0.3, window=0.6)

    def update(self, t, **kwargs):
        return self.tracker.update(t, landmarks(**kwargs), POSE)

    def calibrate(self):
        for n in range(5):
            result = self.update(n / 10)
        return result

    def test_geometry(self):
        sample = measure(landmarks(dx=2))
        self.assertAlmostEqual(sample.ear, 0.3)
        self.assertAlmostEqual(sample.x, 0.1)
        self.assertEqual(sample.y, 0)

    def test_bad_points(self):
        points = landmarks()
        points[468, 0] = np.nan
        self.assertIsNone(measure(points))
        self.assertIsNone(measure(np.zeros((478, 2))))
        self.assertEqual(self.tracker.update(0, points)["tracking_state"], "tracking_failure")
        self.assertEqual(self.tracker.update(0.1, failure=True)["tracking_state"], "tracking_failure")

    def test_calibration_and_window(self):
        self.assertEqual(self.update(0)["tracking_state"], "uncertain")
        self.assertEqual(self.calibrate()["tracking_state"], "valid")
        for n in range(5, 10):
            result = self.update(n / 10)
        self.assertEqual(result["attention"]["score"], 1)
        self.assertEqual(result["gaze"]["dispersion"], 0)
        self.assertGreater(result["gaze"]["stable_seconds"], 0)
        for n in range(10, 18):
            result = self.update(n / 10, dx=3)
        self.assertEqual(result["attention"]["score"], 0)
        self.assertEqual(result["gaze"]["direction"], "right")
        self.assertIsNone(result["gaze"]["screen_x"])

    def test_subthreshold_movement_is_visible_in_continuous_offset(self):
        self.calibrate()
        result = self.update(0.5, dx=0.8)
        self.assertEqual(result["gaze"]["direction"], "center")
        self.assertAlmostEqual(result["gaze"]["center_offset_x"], 0.04)
        missing = self.tracker.update(0.6)
        self.assertIsNone(missing["gaze"]["center_offset_x"])

    def test_calibration_recovers_after_unavailable_pose(self):
        for n in range(10):
            result = self.tracker.update(n / 10, landmarks(), (0, np.nan, 0))
        self.assertEqual(result["eye_calibration_progress"], 0)
        self.assertIsNone(result["head_pose"])
        for n in range(10, 16):
            result = self.update(n / 10)
        self.assertTrue(result["tracking_valid"])

    def test_missing_and_gap(self):
        self.calibrate()
        self.update(0.5, lid=0.5)
        missing = self.tracker.update(0.6)
        self.assertEqual(missing["tracking_state"], "missing_face")
        self.assertIsNone(missing["attention"]["score"])
        self.update(0.7)
        self.assertFalse(self.tracker.blinks)
        self.assertIsNone(self.update(1.9)["attention"]["score"])

    def test_blink_rate_and_closure(self):
        self.calibrate()
        for n in range(5, 111):
            result = self.update(n / 10, lid=0.5 if n in (50, 51) else 3)
            if n == 51:
                self.assertAlmostEqual(result["eyes"]["closure_seconds"], 0.1)
        self.assertEqual(len(self.tracker.blinks), 1)
        self.assertGreater(result["eyes"]["blink_rate_per_minute"], 5)

    def test_reset_and_interrupted_calibration(self):
        self.update(0)
        self.update(0.1)
        self.tracker.update(0.2)
        self.assertLess(self.update(0.3)["eye_calibration_progress"], 0.1)
        self.tracker.reset()
        self.assertEqual(self.tracker.update(0.4)["eye_calibration_progress"], 0)


class AppTests(unittest.TestCase):
    def test_cli(self):
        self.assertEqual(parse_args([]).udp, "127.0.0.1:5005")
        self.assertEqual(parse_args(["record", "calm"]).command, "record")
        self.assertEqual(parse_args(["train"]).command, "train")
        for flag in ("--attention-window", "--smoothing", "--sensitivity"):
            with self.subTest(flag=flag), self.assertRaises(SystemExit):
                parse_args([flag, "nan"])

    @patch("stress_cam.FaceDetector")
    def test_app_packets(self, detector):
        detector.return_value.detect.return_value = Face(landmarks(), {}, POSE)
        app = StressApp(parse_args(["--udp", "off", "--no-pulse", "--baseline"]))
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        for i in range(12):
            app.step(i / 10, frame)
        packet = json.loads(json.dumps(app.message(), allow_nan=False))
        self.assertTrue(packet["face"])
        self.assertIsNone(packet["stress_analysis"]["score"])
        self.assertEqual(packet["time"], 1.1)
        self.assertTrue({"stress", "level", "face", "calibrating", "heart_rate", "blink_rate", "time"} <= packet.keys())
        app.step(1.2, None)
        self.assertFalse(app.message()["face"])
        self.assertEqual(app.message()["tracking_state"], "tracking_failure")
        self.assertFalse(app.message()["stress_analysis"]["behavioral_features"])

    @patch("stress_cam.socket.socket")
    def test_udp_throttle_and_failure(self, new_socket):
        udp = UdpSender("127.0.0.1:5005")
        sock = new_socket.return_value
        sock.sendto.side_effect = OSError
        udp.send(0, {"face": False})
        udp.send(0.01, {"face": False})
        sock.sendto.assert_called_once()
        self.assertEqual(json.loads(sock.sendto.call_args.args[0]), {"face": False})
        udp.close()
        sock.close.assert_called_once()

    def test_udp_loopback(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
            try:
                receiver.bind(("127.0.0.1", 0))
            except PermissionError:
                self.skipTest("Local bind unavailable under sandbox")
            receiver.settimeout(1)
            udp = UdpSender(f"127.0.0.1:{receiver.getsockname()[1]}")
            try:
                udp.send(0, {"stress": None, "face": True})
                self.assertEqual(json.loads(receiver.recv(4096)), {"stress": None, "face": True})
            finally:
                udp.close()
