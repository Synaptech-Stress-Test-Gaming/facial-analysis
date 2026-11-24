# Game API

Live mode sends UTF-8 JSON datagrams to IPv4 UDP `127.0.0.1:5005` at up to 20 Hz. Use `--udp HOST:PORT` to change the destination or `--udp off` to disable publishing. No images are sent. UDP delivery is best effort. For live gameplay, discard old packets or treat the feed as unavailable after a local receive timeout such as 500 ms. Playback runs as fast as decoding and inference permit.

| Field | Meaning |
| --- | --- |
| `schema_version` | 2, with the v1 field names retained. |
| `timestamp`, `time` | Unix wall-clock seconds sampled just after frame acquisition, and monotonic session seconds (clip position for video), respectively. |
| `fps` | Smoothed loop FPS from completed frames, initially zero. Includes I/O and rendering between samples. |
| `face`, `calibrating` | Legacy face-presence and stress baseline flags. Eye calibration has its own progress field. |
| `stress`, `level` | Legacy experimental scalar (0–1) and its existing calm/tense/stressed labels. The last score can persist when the face disappears. |
| `heart_rate`, `blink_rate` | Experimental legacy camera pulse in bpm and blendshape-based blinks/minute, respectively. May reflect the recent window. |
| `tracking_state`, `tracking_valid` | `valid` means calibrated, open eyes with usable pose. `uncertain` covers warm-up, closed eyes, or unavailable pose. `missing_face` is no detected face. `tracking_failure` is failed capture, inference error, or unusable landmark geometry. The boolean is true only for `valid`. |
| `eye_calibration_progress` | Fraction of required center-reference time collected, 0–1. |
| `gaze.direction` | `center`, `left`, `right`, `up`, `down`, or null. Direction is relative to the image passed to the detector (mirrored for webcam and original orientation for video). |
| `gaze.relative_x`, `relative_y` | Mean iris offset from each eye midpoint in eye widths, using the local eye axis and its perpendicular. |
| `gaze.center_offset_x`, `center_offset_y` | Continuous iris offsets from the personal center reference, in eye widths. |
| `gaze.screen_x`, `screen_y`, `on_screen` | Reserved for display calibration. Currently null. |
| `gaze.center_oriented` | Whether iris offsets and head orientation are inside the user's calibrated center band. |
| `gaze.dispersion`, `stable_seconds` | RMS dispersion in eye widths over usable time in the attention window, and seconds inside a tolerance around the most recent stability anchor. |
| `eyes.ear`, `closure_seconds` | Mean eye aspect ratio and elapsed seconds in the current closure. Null closure duration for open or unavailable eyes. |
| `eyes.blink_rate_per_minute` | EAR-based completed blinks per minute of observed time across the last 30 s. Requires at least 10 s of observed time. |
| `head_pose` | Relative-to-camera MediaPipe transform decomposition, in degrees around X, Y, Z (pitch, yaw, roll). Orientation is approximate and depends on image mirroring. |
| `attention.score`, `coverage` | Center-oriented duration / usable duration in the window, and usable duration / window duration. Coverage measures availability. The score requires current valid tracking and at least 50% coverage. |
| `stress_analysis` | Scientific stress score/confidence are null and method is `unvalidated`. Includes the legacy method, its contributions, and the rolling facial and behavioral features when available. |

Use `tracking_valid` with `attention.score` when a game interaction needs center orientation. A zero score describes orientation outside the center band. Players can look elsewhere while engaged. Treat null as unavailable. The behavioral features report mean blendshape values (0–1), head motion (face widths/second), face size (fraction of image width), legacy blink rate, and experimental pulse. Legacy contributions are weighted rule contributions in baseline mode and logistic log-odds contributions in personal-model mode.

## Browser clients

Browser JavaScript needs a bridge for raw UDP. A local process can receive the existing datagrams and forward the latest packet as JSON over a loopback WebSocket. Keep origin checks in the bridge, bind to loopback, and verify the deployed game's secure-context requirements. This release ships the UDP publisher and native receiver example.

## Screen gaze (port 5007)

`screen_gaze.py` is a separate program that sends JSON datagrams to `127.0.0.1:5007` at up to 30 Hz (`--udp HOST:PORT` or `--udp off`). Its gaze object uses the field names reserved above, filled after a dot calibration across the screen.

| Field | Meaning |
| --- | --- |
| `schema_version`, `source` | 2 and `"screen_gaze"`. |
| `timestamp`, `time`, `t_emit` | Unix wall-clock seconds at frame acquisition, monotonic session seconds, and the shared clock at send time, as on port 5005. |
| `face`, `calibrated` | Face detected in this frame; dot calibration finished. |
| `calibration_error_px` | Mean error on left-out calibration dots, in screen pixels. Null before calibration. |
| `head_moved` | The head turned more than 8°, moved sideways more than about a third of a face width, or leaned in or out by 15% since calibration. Accuracy drops; recalibrate. |
| `screen` | `[width, height]` of the screen in pixels used for calibration. |
| `gaze.screen_x`, `screen_y` | Gaze point as a fraction of the screen, 0–1 from the top left, smoothed and clipped to the screen. Held through blinks. Null until calibrated. |
| `gaze.on_screen` | False when the unclipped estimate lies more than 5% outside the screen. |
