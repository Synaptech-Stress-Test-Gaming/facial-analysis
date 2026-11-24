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

On Windows, use `.venv\Scripts\activate` after creating the environment. OpenCV needs permission to use the camera from the terminal or application running Python. `--camera 1` picks another webcam, and setting the environment variable `STRESS_CAMERA=1` makes that the default (on a Mac, camera 0 can be a nearby iPhone through Continuity Camera). MediaPipe downloads its face model on first run if `models/face_landmarker.task` is absent.

Look at the center of the playing area while eye calibration collects open-eye samples. The separate legacy stress baseline runs for 20 seconds by default. Press **c** to recalibrate, **d** for details, or **q** to quit. For playback or game development:

```sh
python stress_cam.py --no-window --no-pulse --profile
python stress_cam.py --video path/to/clip.mp4 --no-window --udp off --profile
python stress_cam.py run --help
python game_receiver_example.py
python -m unittest discover -s tests -v
```

Head tilt, nod, turn and lean are measured from the neutral posture learned during the 20-second calibration, so sitting normally reads 0, and are sent as `head_movement` for use as game input; see the [API](docs/api.md#head-movement). Add `--save-json` to also save every frame's values to `recordings/` as JSON lines for later analysis. Default UDP destination is `127.0.0.1:5005`. `--eye-calibration` and `--attention-window` accept durations in seconds. The calibrated gaze reference is a center target. Display coordinates and screen-presence classification are unavailable. Camera frames stay local.

## Existing personal model

```sh
python stress_cam.py record calm
python stress_cam.py record stressed
python stress_cam.py train
```

These commands retain the historical feature columns, recordings and model format. Recordings are CSVs of derived features. Existing labels are supplied by the person running the session. `train` performs session-level cross-validation when enough sessions are available. Live mode uses the saved model when present. Use `--baseline` to select the historical calibration heuristic. Treat the legacy score as an experimental game input. The new scientific stress score is unavailable.

## Muse 2 EEG viewer

```sh
python muse_eeg.py
python muse_eeg.py --synthetic
```

`muse_eeg.py` connects to a Muse 2 over Bluetooth through BrainFlow and shows the four electrodes (TP9, AF7, AF8, TP10) filtered to 1–45 Hz, the skin contact of each, the share of power in each brainwave band over the last 2 s, and an EEG stress index: beta / (alpha + theta) compared with a 20 s relaxed calibration. Close the Muse app on your phone first, because the headband talks to one device at a time. `--name Muse-XXXX` picks one headband and `--synthetic` uses BrainFlow's simulated board. Values go to UDP `127.0.0.1:5006` with `"source": "eeg"`.

## Screen gaze

```sh
python screen_gaze.py
python screen_gaze.py --points 9
```

`screen_gaze.py` estimates where on the screen you look, which the centre-target eye calibration in `stress_cam.py` leaves null. It runs full screen: check that the camera preview says "Eyes found", press space, keep your head still and look at each red dot until it disappears (13 dots, about 25 s; `--points 9` or `16`). The result screen shows the accuracy measured on dots the model did not train on. Then a dot follows your gaze and the 3x3 grid cell you look at lights up. Press **c** to recalibrate, **g** for the grid, **p** for the camera preview, or **q** to quit.

It records `eye_tracking.measure()` and MediaPipe's gaze blendshapes at each dot and fits a ridge regression from them to screen position, choosing the model by leave-one-dot-out cross-validation. Expect a few centimetres of error: enough for a region of the screen, not for single words. Moving your head after calibrating makes it worse, and the app warns when you do. Gaze goes to UDP `127.0.0.1:5007`; see the [API](docs/api.md#screen-gaze-port-5007).
