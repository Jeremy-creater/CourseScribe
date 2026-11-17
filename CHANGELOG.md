# Changelog

[中文说明见下方](#中文)

## [0.1.0] - 2026-11-17

First public release. Windows 11 and macOS (Apple silicon); the user interface is in Chinese.

Download `CourseScribe-0.1.0-windows-x64.zip` or `CourseScribe-0.1.0-macos-arm64.zip` from the Release, unzip it and run it: no Python needed. Models are not included: you need [Ollama](https://ollama.com) with `gemma4:12b`, and the program asks before it downloads the transcription model (about 3.1 GB). The apps are not code-signed, so Windows SmartScreen and macOS Gatekeeper warn on the first start; the README explains how to open them.

### Added

- Record the sound your computer plays (no microphone, no virtual sound card), or pick or drag in audio and video files.
- A raw transcript (`.txt`) and a proofread transcript (`.md`) that only removes filler words, tics and immediate repeats; anything else removed is detected and retried.
- Optional study notes (`-整理笔记.md`) from a second local model, with quotes checked word for word against the transcript.
- A global hotkey you can change, a word list for mishearings, editable notes rules.
- Tasks are saved after every step and resume after the program is closed or crashes; clear reasons and a way to continue for a busy file, a full disk, sleep, a changed output device, a stopped Ollama.
- Keeps the computer awake while it works; nothing is downloaded without asking; audio and text never leave the computer.

### Known limits

- Tested on Windows 11 with an RTX 5080 and on a Mac with an M1 Pro only. Windows 10 and Intel Macs are untested.
- On a Mac the transcription runs on the processor, at about 0.8x real time.
- A very long text can make the notes model hit its answer length limit; the program says so and you can retry.

---

## 中文

## [0.1.0] - 2026-11-17

第一个公开版本。支持 Windows 11 和 macOS（Apple 芯片），界面只有中文。

从 Release 下载 `CourseScribe-0.1.0-windows-x64.zip` 或 `CourseScribe-0.1.0-macos-arm64.zip`，解压后运行，不需要装 Python。模型不在包里：需要安装 [Ollama](https://ollama.com) 并拉取 `gemma4:12b`，转录模型（约 3.1 GB）下载前程序会先问你。程序没有代码签名，第一次打开 Windows 和 macOS 都会警告，README 里写了怎么打开。

### 新增

- 录下电脑播放的声音（不录麦克风，不需要虚拟声卡），或选择、拖入音视频文件。
- 原始文字稿（`.txt`）和校对稿（`.md`）：校对只删语气词、口头禅和紧挨着的重复，其他删减会被检测出来并重试。
- 可选的学习笔记（`-整理笔记.md`），由第二个本地模型整理，引述和校对稿逐字核对。
- 可修改的全局快捷键、听错词表、可编辑的整理规则。
- 每完成一步保存任务，程序关闭或崩溃后接着做；文件被占用、磁盘满、睡眠、换输出设备、Ollama 没启动时，都有明确的原因和继续的办法。
- 处理期间不让电脑睡眠；不经询问不下载任何东西；音频和文字不离开电脑。

### 已知限制

- 只在装有 RTX 5080 的 Windows 11 和 M1 Pro 的 Mac 上测过，Windows 10 和 Intel 芯片的 Mac 没测过。
- Mac 上转录在处理器上跑，约 0.8 倍实时。
- 很长的文字稿可能让笔记模型的回答写到长度上限，程序会说明，可以重试。
