# CourseScribe

Record a course while you watch it, or pick a local audio or video file, and get a full proofread transcript. CourseScribe keeps the original wording: it does not summarize. Everything runs on your own computer.

Under development. Windows works today; Mac recording is planned next. Requirements are in [docs/PRD.md](docs/PRD.md).

## Run from source (Windows)

You need Python 3.12 and [Ollama](https://ollama.com) with the proofreading model:

```text
ollama pull gemma4:12b
python -m pip install -e .[dev,gpu]
python -m coursescribe
```

The first task asks before downloading the transcription model (about 3.1 GB).

## Build the Windows package

```text
python -m PyInstaller packaging/coursescribe.spec --noconfirm
```

The program appears in `dist/CourseScribe`. Models are not included.
