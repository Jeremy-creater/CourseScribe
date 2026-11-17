# CourseScribe

Record an online course while you watch it, or pick a local audio or video file, and get a full proofread transcript. CourseScribe keeps the original wording: it does not summarize. Everything runs on your own computer; audio and text never leave it. The only network use is downloading the transcription model, and it asks first.

## What you get

For each recording or file:

- `name.txt`: the raw transcript. Nothing edits it after it is written.
- `name.md`: the proofread version. The proofreading model only removes filler words, verbal tics and immediate repeats, and fixes typos, punctuation and paragraphs. It does not rewrite or shorten.
- Optional: `name-notes.md`, study notes written by a local model from the proofread text. It is a separate file and never touches the other two.

Other features: a global shortcut to start and stop recording (Ctrl+Alt+R by default on Windows, Control+Option+R on Mac; it can be changed in Settings), a word list for terms the model mishears, and tasks that resume after the program is closed. The interface is in Chinese. Requirements are in [docs/PRD.md](docs/PRD.md) (Chinese).

## What you need

- Windows 11 or macOS 14+ (Apple silicon). Recording the computer's sound needs macOS 14.2+.
- [Ollama](https://ollama.com) running locally, with the proofreading model:

  ```text
  ollama pull gemma4:12b
  ollama pull qwen3.5:9b   # only for the notes feature
  ```

- About 3.1 GB of disk for the transcription model (Whisper `large-v3`), downloaded on the first task after you agree.
- An NVIDIA GPU makes transcription much faster on Windows. On a Mac it runs on the CPU: about 0.8x real time on an M1 Pro, so a one-hour class takes roughly 48 minutes.

## Run from source

You need Python 3.12.

```text
python -m pip install -e .[dev,gpu]    # on a Mac use .[dev]
python -m coursescribe
```

## Build the app

```text
python -m PyInstaller packaging/coursescribe.spec --noconfirm
```

On Windows the program appears in `dist/CourseScribe` (run `CourseScribe.exe`; keep the `_internal` folder next to it). On a Mac it is `dist/CourseScribe.app`. Models are not included. The builds are not code-signed, so Windows SmartScreen or macOS Gatekeeper may warn on the first start.

## Tests

```text
python -m pytest -m "not engine and not integration and not gui and not perf"
```

The other groups need a real model, a running Ollama or a real desktop; see `CLAUDE.md`.

## License

MIT, see [LICENSE](LICENSE). Bundled third-party software is listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
