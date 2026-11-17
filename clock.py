"""The one clock every recorded stream shares.

Frame timestamps, the UDP JSON, the LSL StreamOutlet and the recording CSVs all call
now() for their timestamps, so a session recorded alongside the Muse 2 EEG in
LabRecorder lines up without any extra offset correction.
"""

from __future__ import annotations

import pylsl


def now() -> float:
    """Seconds on LSL's shared clock (pylsl.local_clock())."""
    return pylsl.local_clock()
