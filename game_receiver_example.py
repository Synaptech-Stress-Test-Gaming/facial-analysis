"""What the game side looks like: read stress values sent by stress_cam.py over UDP.

Start stress_cam.py, then run this in a second terminal. A real game calls
latest_message() once per frame and maps `stress` onto a mechanic.
"""

import json
import socket
import time

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("127.0.0.1", 5005))
sock.setblocking(False)  # never let the game freeze while waiting for a message


def latest_message():
    """Newest message waiting in the socket, or None if nothing new arrived."""
    message = None
    while True:
        try:
            message = json.loads(sock.recv(4096))
        except BlockingIOError:
            return message


print("Listening on 127.0.0.1:5005 (Ctrl+C to stop) ...")
stress, calibrating = 0.0, False
try:
    while True:  # stand-in for the game loop
        message = latest_message()
        if message:
            calibrating = message["calibrating"]
            if message["stress"] is not None:
                stress = message["stress"]
        jump_power = 1.0 + 0.5 * stress  # e.g. jump up to 50% higher when the player is stressed
        state = "calibrating" if calibrating else f"stress {stress:.2f}"
        print(f"{state:<14} -> jump x{jump_power:.2f}   ", end="\r")
        time.sleep(1 / 30)
except KeyboardInterrupt:
    print()
