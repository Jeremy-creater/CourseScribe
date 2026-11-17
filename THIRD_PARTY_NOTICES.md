# Third-party software

CourseScribe's own code is MIT-licensed (see `LICENSE`). The packaged app also contains the
software below, each under its own license. This list was written from the installed packages'
metadata; check the package pages for the full license texts.

| Component | License | Used for |
|---|---|---|
| PySide6-Essentials / Shiboken6 (Qt for Python) | LGPL-3.0 (or GPL) | the window |
| faster-whisper | MIT | transcription |
| CTranslate2 | MIT | transcription engine |
| ONNX Runtime | MIT | voice detection |
| PyAV (`av`) | BSD-3-Clause | reading audio and video files |
| SoundCard (Windows) | BSD-3-Clause | recording the computer's sound |
| catap (Mac) | see its package page | recording the computer's sound |
| NumPy | BSD-3-Clause and others | numbers |
| platformdirs | MIT | default folders |
| huggingface-hub | Apache-2.0 | downloading the transcription model |
| tokenizers | Apache-2.0 | transcription model |
| PyInstaller (its bootloader is inside the packaged app) | GPL-2.0 with a special exception that allows the programs it builds under any license | packaging |
| Python runtime (inside the packaged app) | PSF license | running the program |
| PyObjC (Mac app only) | MIT | talking to macOS |
| NVIDIA cuBLAS and cuDNN (Windows package only) | NVIDIA proprietary license | running the transcription model on an NVIDIA GPU |

## NVIDIA libraries (Windows package)

The Windows package contains NVIDIA's cuBLAS and cuDNN runtime libraries so that transcription can use an NVIDIA GPU. They are distributed under NVIDIA's license terms, and the full texts are in the `licenses` folder of the package (`nvidia-cublas-cu12` and `nvidia-cudnn-cu12`). cuBLAS names its DLLs as distributable in its license. The cuDNN license lists its distributable files by older names, so whether the 9.x DLLs are covered word for word is a judgment call; if NVIDIA or a reader objects, the package can be rebuilt without them (transcription then runs on the processor).

## Qt (LGPL)

Qt is used as shared libraries that sit as separate files inside the app folder (`_internal` on
Windows, inside the `.app` on Mac). You may replace them with your own build of the same Qt for
Python version. The Qt for Python license texts are at <https://doc.qt.io/qtforpython-6/licenses.html>.

## Models are not included

The app does not ship any model. With your consent it downloads the Whisper `large-v3` model from
Hugging Face, and it talks to a local [Ollama](https://ollama.com) for proofreading and notes. Those
models have their own licenses; read them before you rely on them.
