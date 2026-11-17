"""FastAPI server around stress_cam.py, for games that want HTTP/WebSocket instead of UDP.

  python server.py                      serve on 127.0.0.1:8000, with the OpenCV window
  python server.py --headless           no OpenCV window (e.g. on a server box)
  python server.py --port 9000 --camera 1
  python server.py --debug-frame        also save debug_frame.jpg once a second

The webcam capture/scoring loop (stress_cam.StressApp) runs on the MAIN thread, exactly
like `python stress_cam.py` does: on macOS, Cocoa wants cv2.namedWindow/imshow/waitKey
(and, we've seen in practice, camera frame delivery itself) on the main thread, so
running it from a background thread does not reliably work there. uvicorn instead runs
in a background thread; that is the officially supported way to drive it programmatically
(uvicorn.Server.install_signal_handlers() no-ops itself off the main thread).

The FastAPI handlers never mutate StressApp directly. They read a shared, lock-protected
metrics dict that the capture loop publishes once per frame, and send it commands
(calibrate / label / record start-stop) through a queue that the capture loop drains
between frames via StressApp.run()'s on_step hook. The UDP output in stress_cam.py
(message()/UdpSender) is untouched and keeps working side by side with this server.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import queue
import sys
import threading
import time
import traceback

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

import stress_cam
from stress_cam import StressApp
from stress_model import BaselineScorer

STALE_AFTER_SECONDS = 2.0  # /health calls capture_running False if no frame this recent


class Shared:
    """The only state the FastAPI handlers read. metrics/last_frame_wall/fps are written
    once per frame from the capture loop (the main thread); capture_alive/last_error are
    written once, when that loop exits. Request handlers never mutate any of it.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.metrics: dict | None = None
        self.last_frame_wall: float | None = None
        self.fps: float = 0.0
        self.capture_alive: bool = False
        self.last_error: str | None = None


shared = Shared()
commands: queue.Queue[tuple[str, object, dict]] = queue.Queue()


def _build_metrics(app: StressApp) -> dict:
    data = app.message()  # unchanged UDP schema; see stress_cam.StressApp.message
    data["eye"] = app.eye_cues()
    data["label"] = app.label
    data["recording"] = app.label_recorder.path.name if app.label_recorder else None
    return data


def _drain_commands(app: StressApp) -> None:
    """Applies queued HTTP-triggered commands, on the capture thread, between frames."""
    while True:
        try:
            cmd, payload, box = commands.get_nowait()
        except queue.Empty:
            return
        try:
            if cmd == "calibrate":
                app.recalibrate()
                result = {"ok": isinstance(app.scorer, BaselineScorer), "calibrating": not app.scorer.ready}
            elif cmd == "label":
                app.set_label(payload)
                result = {"ok": True, "label": app.label}
            elif cmd == "record_start":
                app.start_recording(payload)
                result = {"ok": True, "path": app.label_recorder.path.name}
            elif cmd == "record_stop":
                result = {"ok": True, **app.stop_recording()}
            else:
                result = {"ok": False, "error": f"unknown command {cmd!r}"}
        except Exception as e:  # a bad request must not take down the capture loop
            result = {"ok": False, "error": str(e)}
        box["result"] = result
        box["event"].set()


def _publish(app: StressApp) -> None:
    now = time.perf_counter()
    with shared.lock:
        if shared.last_frame_wall is not None:
            rate = 1.0 / max(now - shared.last_frame_wall, 1e-6)
            shared.fps = 0.9 * shared.fps + 0.1 * rate if shared.fps else rate
        shared.last_frame_wall = now
        shared.metrics = _build_metrics(app)


def _save_debug_frame(app: StressApp, state: dict) -> None:
    """Throttled to once a second; logs shape and mean brightness, saves debug_frame.jpg."""
    if app.frame is None or app.t - state["last"] < 1.0:
        return
    state["last"] = app.t
    frame: np.ndarray = app.frame
    mean_brightness = float(frame.mean())
    cv2.imwrite("debug_frame.jpg", frame)
    print(f"[debug-frame] shape={frame.shape} mean_brightness={mean_brightness:.1f}")


def capture_loop(args: argparse.Namespace, debug_frame: bool) -> None:
    """Runs on the main thread. Blocks until the window is closed, the video ends, or
    the webcam disconnects (StressApp.run raises SystemExit in that case)."""
    with shared.lock:
        shared.capture_alive = True
    app = StressApp(args)
    debug_state = {"last": -1.0}
    try:
        def on_step():
            _drain_commands(app)
            _publish(app)
            if debug_frame:
                _save_debug_frame(app, debug_state)

        app.run(on_step=on_step)
    except BaseException as e:  # SystemExit on webcam disconnect must be logged, not silent
        with shared.lock:
            shared.last_error = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        with shared.lock:
            shared.capture_alive = False


def send_command(cmd: str, payload: object, timeout: float = 5.0) -> dict:
    box = {"event": threading.Event()}
    commands.put((cmd, payload, box))
    if not box["event"].wait(timeout):
        raise HTTPException(status_code=504, detail="capture loop did not respond in time")
    return box["result"]


app = FastAPI(title="Facial analysis stress server")


class RecordStartRequest(BaseModel):
    name: str = "session"


@app.get("/health")
def health() -> dict:
    now = time.perf_counter()
    with shared.lock:
        capture_alive = shared.capture_alive
        last_frame_wall = shared.last_frame_wall
        fps = shared.fps
        last_error = shared.last_error
        data = shared.metrics
    last_frame_age = None if last_frame_wall is None else now - last_frame_wall
    fresh = last_frame_age is not None and last_frame_age < STALE_AFTER_SECONDS
    return {
        "capture_running": capture_alive and fresh,
        "face_found": bool(data["face"]) if data else False,
        "last_frame_age_seconds": None if last_frame_age is None else round(last_frame_age, 2),
        "fps": round(fps, 1),
        "last_error": last_error,
    }


@app.get("/metrics")
def metrics() -> dict:
    with shared.lock:
        data = shared.metrics
    if data is None:
        raise HTTPException(status_code=503, detail="capture loop hasn't produced a frame yet")
    return data


@app.post("/calibrate")
def calibrate() -> dict:
    return send_command("calibrate", None)


@app.post("/label/{name}")
def label(name: str) -> dict:
    return send_command("label", name)


@app.post("/record/start")
def record_start(body: RecordStartRequest) -> dict:
    return send_command("record_start", body.name)


@app.post("/record/stop")
def record_stop() -> dict:
    return send_command("record_stop", None)


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    """Pushes the latest metrics every 20 ms (50 Hz), same payload as GET /metrics."""
    await websocket.accept()
    try:
        while True:
            with shared.lock:
                data = shared.metrics
            if data is not None:
                await websocket.send_json(data)
            await asyncio.sleep(0.02)
    except WebSocketDisconnect:
        pass


def _build_cam_args(ns: argparse.Namespace) -> argparse.Namespace:
    argv = ["run", "--camera", str(ns.camera), "--calibration", str(ns.calibration),
            "--sensitivity", str(ns.sensitivity), "--smoothing", str(ns.smoothing), "--udp", ns.udp]
    if ns.video:
        argv += ["--video", ns.video]
    if ns.no_pulse:
        argv.append("--no-pulse")
    if ns.baseline:
        argv.append("--baseline")
    if ns.headless:
        argv.append("--no-window")
    return stress_cam.parse_args(argv)


def _parse_server_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--headless", action="store_true", help="no OpenCV window (e.g. on a headless box)")
    parser.add_argument("--camera", type=int, default=0, help="webcam number (default 0)")
    parser.add_argument("--video", help="read a video file instead of the webcam")
    parser.add_argument("--no-pulse", action="store_true", help="turn off the experimental heart-rate estimate")
    parser.add_argument("--baseline", action="store_true", help="ignore your personal model and calibrate instead")
    parser.add_argument("--calibration", type=float, default=20.0, help="calibration length in seconds (default 20)")
    parser.add_argument("--sensitivity", type=float, default=0.4, help="baseline stress sensitivity (default 0.4)")
    parser.add_argument("--smoothing", type=float, default=1.0, help="seconds of smoothing on the output (default 1)")
    parser.add_argument("--udp", default="127.0.0.1:5005", help="also send UDP here, or 'off' (default 127.0.0.1:5005)")
    parser.add_argument("--debug-frame", action="store_true",
                         help="save debug_frame.jpg and log its shape/mean brightness once a second")
    return parser.parse_args(argv)


def main() -> None:
    ns = _parse_server_args(sys.argv[1:])
    cam_args = _build_cam_args(ns)

    import uvicorn

    config = uvicorn.Config(app, host=ns.host, port=ns.port, workers=1, reload=False)
    uv_server = uvicorn.Server(config)

    server_thread = threading.Thread(target=uv_server.run, daemon=True, name="uvicorn")
    server_thread.start()

    print(f"Serving on http://{ns.host}:{ns.port} (docs at /docs). Single worker, no --reload.")
    print("Capture loop (and the preview window, unless --headless) runs on this main thread.")
    try:
        capture_loop(cam_args, debug_frame=ns.debug_frame)
    finally:
        uv_server.should_exit = True
        server_thread.join(timeout=5.0)


if __name__ == "__main__":
    main()
