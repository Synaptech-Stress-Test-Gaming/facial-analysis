import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import head_movement
from face_features import Face
from head_movement import HeadMovement
from stress_cam import StressApp, parse_args

FPS = 30
SETTLE = head_movement.SETTLE_SECONDS


def rotation(pitch=0.0, yaw=0.0, roll=0.0):
    """Rotation matrix whose cv2.RQDecomp3x3 angles are (pitch, yaw, roll), like MediaPipe's pose."""
    def axis(a, i, j):
        r, (c, s) = np.eye(3), (math.cos(math.radians(a)), math.sin(math.radians(a)))
        r[i, i], r[i, j], r[j, i], r[j, j] = c, -s, s, c
        return r
    return axis(roll, 0, 1) @ axis(yaw, 2, 0) @ axis(pitch, 1, 2)


def face(pitch=0.0, yaw=0.0, roll=0.0, distance=50.0, matrix=None):
    """A face with a head pose (MediaPipe angles in degrees) at a distance from the camera (cm)."""
    transform = np.eye(4)
    transform[:3, :3] = rotation(pitch, yaw, roll) if matrix is None else matrix
    transform[:3, 3] = (0.0, 0.0, -distance)
    return Face(np.zeros((478, 2), np.float32), {"jawOpen": 0.1}, (pitch, yaw, roll), transform)


def run(head, t, seconds, rng=None, jitter=0.0, **pose):
    """Feeds `seconds` of frames at 30 fps (with optional landmark jitter, degrees); returns
    the time and the last result."""
    result = None
    for _ in range(round(seconds * FPS)):
        t += 1 / FPS
        noisy = {k: pose.get(k, 0.0) + (rng.normal(0, jitter) if jitter else 0.0) for k in ("pitch", "yaw", "roll")}
        result = head.update(t, face(**noisy, distance=pose.get("distance", 50.0)))
    return t, result


def calibrated_head(seconds=2.0):
    head = HeadMovement(calibration=seconds)
    t, result = run(head, 0.0, SETTLE + seconds + 0.3)
    assert result["calibrated"]
    return head, t


class CalibrationTest(unittest.TestCase):
    def test_progress_and_reset(self):
        head = HeadMovement(calibration=2.0)
        t, result = run(head, 0.0, SETTLE + 1.0)
        self.assertFalse(result["calibrated"])
        self.assertAlmostEqual(result["calibration_progress"], 0.5, delta=0.05)
        self.assertIsNone(result["tilt"])
        t, result = run(head, t + 5.0, 0.5)  # a 5 s gap doesn't count as calibration time
        self.assertLess(result["calibration_progress"], 0.8)
        t, result = run(head, t, 1.0)
        self.assertTrue(result["calibrated"])
        head.reset()
        self.assertFalse(head.update(t + 0.1, face())["calibrated"])

    def test_neutral_reads_zero_despite_jitter(self):
        rng = np.random.default_rng(0)
        head = HeadMovement(calibration=4.0)
        t, result = run(head, 0.0, SETTLE + 5.0, rng, jitter=0.4, roll=3.0, pitch=-8.0, yaw=5.0)
        self.assertTrue(result["calibrated"])  # jitter alone isn't "moving"
        _, result = run(head, t, 1.0, roll=3.0, pitch=-8.0, yaw=5.0)
        for name in ("tilt", "nod", "turn"):
            self.assertAlmostEqual(result[name]["degrees"], 0.0, delta=0.5)
            self.assertEqual(result[name]["state"], "none")
        self.assertAlmostEqual(result["lean"]["fraction"], 0.0, delta=0.005)

    def test_settling_in_is_ignored(self):
        head = HeadMovement(calibration=2.0)
        t, _ = run(head, 0.0, SETTLE - 0.1, roll=-30.0, distance=40.0)  # still getting comfortable
        t, _ = run(head, t, 3.0)
        _, result = run(head, t, 0.5)
        self.assertAlmostEqual(result["tilt"]["degrees"], 0.0, delta=0.5)
        self.assertAlmostEqual(result["lean"]["fraction"], 0.0, delta=0.005)

    def test_moving_during_calibration_is_ignored(self):
        head = HeadMovement(calibration=4.0)
        t, _ = run(head, 0.0, SETTLE + 1.5)
        for i in range(FPS):  # a 1 s look around: 60 degrees/s, too fast to count
            t += 1 / FPS
            head.update(t, face(yaw=60 * math.sin(2 * math.pi * i / FPS)))
        t, _ = run(head, t, 1.0, roll=-20.0)  # a short, still tilt: a minority, so the median ignores it
        t, result = run(head, t, 3.0)
        self.assertTrue(result["calibrated"])
        _, result = run(head, t, 0.5)
        self.assertAlmostEqual(result["tilt"]["degrees"], 0.0, delta=0.5)
        self.assertAlmostEqual(result["turn"]["degrees"], 0.0, delta=0.5)

    def test_bad_calibration_length(self):
        for value in (0, -1, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                HeadMovement(value)


class MovementTest(unittest.TestCase):
    def setUp(self):
        self.head, self.t = calibrated_head()

    def test_directions(self):
        # MediaPipe angles in the mirrored image: +pitch looks down, +yaw turns to the
        # player's right, +roll tilts toward the player's left shoulder.
        cases = (({"roll": 20}, "tilt", "left", -20), ({"roll": -20}, "tilt", "right", 20),
                 ({"pitch": 18}, "nod", "down", -18), ({"pitch": -18}, "nod", "up", 18),
                 ({"yaw": 25}, "turn", "right", 25), ({"yaw": -25}, "turn", "left", -25))
        for pose, name, state, degrees in cases:
            with self.subTest(pose=pose):
                _, result = run(self.head, self.t, 1.0, **pose)
                self.assertEqual(result[name]["state"], state)
                self.assertAlmostEqual(result[name]["degrees"], degrees, delta=0.5)
                others = {"tilt", "nod", "turn", "lean"} - {name}
                self.assertTrue(all(result[o]["state"] == "none" for o in others))
                self.t, _ = run(self.head, self.t + 1.0, 1.0)  # back to neutral

    def test_camera_above_the_screen(self):
        # Neutral posture pitched 25 degrees relative to the camera, then a pure head turn.
        neutral = rotation(pitch=25.0)
        head = HeadMovement(calibration=1.0)
        t = 0.0
        for _ in range(round((SETTLE + 1.3) * FPS)):
            t += 1 / FPS
            head.update(t, face(matrix=neutral))
        for _ in range(FPS):
            t += 1 / FPS
            result = head.update(t, face(matrix=neutral @ rotation(yaw=30.0)))
        self.assertAlmostEqual(result["turn"]["degrees"], 30.0, delta=0.5)
        self.assertAlmostEqual(result["tilt"]["degrees"], 0.0, delta=0.5)
        self.assertAlmostEqual(result["nod"]["degrees"], 0.0, delta=0.5)

    def test_lean(self):
        _, result = run(self.head, self.t, 1.0, distance=40.0)  # 10 cm closer
        self.assertEqual(result["lean"]["state"], "forward")
        self.assertAlmostEqual(result["lean"]["fraction"], 0.2, delta=0.01)
        self.assertAlmostEqual(result["lean"]["cm"], 10.0, delta=0.5)
        self.assertAlmostEqual(result["lean"]["amount"], 0.8, delta=0.04)
        _, result = run(self.head, self.t + 2.0, 1.0, distance=60.0)
        self.assertEqual(result["lean"]["state"], "back")

    def test_amount_is_clipped(self):
        _, result = run(self.head, self.t, 1.0, yaw=80)
        self.assertEqual(result["turn"]["amount"], 1.0)

    def test_hysteresis(self):
        t, result = run(self.head, self.t, 0.5, roll=-13)
        self.assertEqual(result["tilt"]["state"], "right")
        t, result = run(self.head, t, 0.5, roll=-10)  # below "enter", above "leave": stays
        self.assertEqual(result["tilt"]["state"], "right")
        t, result = run(self.head, t, 0.5, roll=-6)
        self.assertEqual(result["tilt"]["state"], "none")

    def test_reacts_within_two_frames(self):
        t, frames = self.t, 0
        while True:
            t += 1 / FPS
            frames += 1
            if self.head.update(t, face(roll=-25))["tilt"]["state"] == "right":
                break
        self.assertLessEqual(frames, 2)  # about 67 ms at 30 fps

    def test_face_lost(self):
        result = self.head.update(self.t + 1 / FPS, None)
        self.assertTrue(result["calibrated"])
        self.assertIsNone(result["tilt"])
        json.dumps(result, allow_nan=False)


class AppTest(unittest.TestCase):
    @patch("stress_cam.FaceDetector")
    def test_message_and_json_log(self, detector):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as folder, patch("stress_cam.RECORDINGS", Path(folder)):
            app = StressApp(parse_args(["--udp", "off", "--no-pulse", "--baseline", "--save-json",
                                        "--head-calibration", "0.5"]))
            neutral = round((SETTLE + 0.8) * FPS)
            detector.return_value.detect.return_value = face()
            for i in range(neutral):
                app.step(i / FPS, frame)
            detector.return_value.detect.return_value = face(roll=-20)
            for i in range(neutral, neutral + 20):
                app.step(i / FPS, frame)
            packet = json.loads(json.dumps(app.message(), allow_nan=False))
            self.assertEqual(packet["head_movement"]["tilt"]["state"], "right")
            app.json_log.close()
            lines = app.json_log.path.read_text().splitlines()
            self.assertEqual(len(lines), neutral + 20)  # every frame, not only the 20 Hz UDP ones
            last = json.loads(lines[-1])
            self.assertEqual(last["head_movement"]["tilt"]["state"], "right")
            self.assertEqual(last["blendshapes"], {"jawOpen": 0.1})

    @patch("stress_cam.FaceDetector")
    def test_calibrates_with_the_stress_baseline(self, detector):
        app = StressApp(parse_args(["--udp", "off", "--no-pulse", "--baseline", "--calibration", "12"]))
        self.assertEqual(app.head_movement.calibration, 12.0)

    def test_cli(self):
        self.assertIsNone(parse_args([]).head_calibration)
        self.assertEqual(parse_args(["--head-calibration", "3"]).head_calibration, 3.0)
        self.assertFalse(parse_args([]).save_json)
        with self.assertRaises(SystemExit):
            parse_args(["--head-calibration", "0"])


if __name__ == "__main__":
    unittest.main()
