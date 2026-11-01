"""Records until killed, so a test can kill it the hard way."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from coursescribe.platform import open_system_audio
from coursescribe.recorder import RecordingSession

session = RecordingSession(open_system_audio, Path(sys.argv[1]))
session.start()
print("recording", flush=True)
time.sleep(600)
