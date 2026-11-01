"""Make speech samples on the fly with the system's own voice; none are stored."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if not found:
        pytest.fail("ffmpeg is needed to make test samples: install it and add it to PATH")
    return found


def speak_to_wav(text: str, target: Path) -> Path:
    """Write ``text`` as spoken Chinese to a WAV file."""
    if sys.platform == "win32":
        script = (
            "[Console]::InputEncoding = [Text.Encoding]::UTF8;"
            "Add-Type -AssemblyName System.Speech;"
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
            "$s.SelectVoiceByHints('NotSet','NotSet',0,[Globalization.CultureInfo]'zh-CN');"
            f"$s.SetOutputToWaveFile('{target}');"
            "$s.Speak([Console]::In.ReadToEnd()); $s.Dispose()"
        )
        command = ["powershell.exe", "-NoProfile", "-Command", script]
        done = subprocess.run(command, input=text.encode("utf-8"), capture_output=True, timeout=300)
        if done.returncode or not target.exists():
            pytest.fail(f"Could not make speech (is a Chinese voice installed?): {done.stderr.decode('utf-8', 'replace')}")
    else:
        temp = target.with_suffix(".aiff")
        subprocess.run(["say", "-v", "Tingting", "-o", str(temp), text], check=True, timeout=300)
        convert(temp, target)
        temp.unlink()
    return target


def convert(source: Path, target: Path, *options: str) -> Path:
    command = [_ffmpeg(), "-y", "-loglevel", "error", *options, "-i", str(source), str(target)]
    subprocess.run(command, check=True, timeout=300)
    return target


def as_video(source: Path, target: Path) -> Path:
    """Wrap the audio in a video with a black picture."""
    command = [
        _ffmpeg(), "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=black:s=160x120:r=5",
        "-i", str(source), "-shortest", "-c:v", "libx264", "-c:a", "aac", str(target),
    ]
    subprocess.run(command, check=True, timeout=300)
    return target
