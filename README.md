# Facial Analysis for games

Runs local webcam facial analysis and publishes eye geometry, a center-orientation proxy, and experimental legacy stress fields over UDP. See the [API](docs/api.md), [architecture and evaluation protocol](docs/architecture.md), and [research review](docs/research_review.md).

## Install and run

Use Python 3.12 and a native Python interpreter on Apple Silicon.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python stress_cam.py
```

On Windows, use `.venv\Scripts\activate` after creating the environment. OpenCV needs permission to use the camera from the terminal or application running Python. MediaPipe downloads its face model on first run if `models/face_landmarker.task` is absent.

Look at the center of the playing area while eye calibration collects open-eye samples. The separate legacy stress baseline runs for 20 seconds by default. Press **c** to recalibrate, **d** for details, or **q** to quit. For playback or game development:

```sh
python stress_cam.py --no-window --no-pulse --profile
python stress_cam.py --video path/to/clip.mp4 --no-window --udp off --profile
python stress_cam.py run --help
python game_receiver_example.py
python -m unittest discover -s tests -v
```

Default UDP destination is `127.0.0.1:5005`. `--eye-calibration` and `--attention-window` accept durations in seconds. The calibrated gaze reference is a center target. Display coordinates and screen-presence classification are unavailable. Camera frames stay local.

## Existing personal model

```sh
python stress_cam.py record calm
python stress_cam.py record stressed
python stress_cam.py train
```

These commands retain the historical feature columns, recordings and model format. Recordings are CSVs of derived features. Existing labels are supplied by the person running the session. `train` performs session-level cross-validation when enough sessions are available. Live mode uses the saved model when present. Use `--baseline` to select the historical calibration heuristic. Treat the legacy score as an experimental game input. The new scientific stress score is unavailable.
