CourseScribe — 录下网课声音或选音视频文件，得到文字稿和校对稿，全程在本机
==========================================================================

使用（Apple 芯片，macOS 14.2 及以上）
  1. 先安装并启动 Ollama（https://ollama.com），然后在终端运行：ollama pull gemma4:12b
     （要用“整理笔记”再运行：ollama pull qwen3.5:9b）
  2. 把 CourseScribe.app 拖进“应用程序”，双击打开。
  3. 第一次被系统拦住：到“系统设置 → 隐私与安全性”，找到 CourseScribe，点“仍要打开”。
     这是因为程序没有苹果开发者签名，不是病毒。
  4. 第一次录音时系统会问是否允许录制系统声音，点允许。换新版本后可能要重新授权。
  5. 点“开始录音”（或按 ⌃⌥R）照常看课，看完点“停止”。也可以把音视频文件拖进窗口。
  6. 第一个任务会问你是否下载转录模型（约 3.1 GB），同意后才下载。Mac 上转录在处理器上跑，
     大约 0.8 倍实时，1 小时的课要约 48 分钟。

结果默认在“文档”里的 CourseScribe 文件夹（文稿被 iCloud 同步时是主文件夹下的 CourseScribe）。
文件位置：设置、任务状态、日志在 ~/Library/Application Support/CourseScribe。
卸载：把 app 移到废纸篓，再删除上面的设置文件夹。

完整说明（中文）https://github.com/Jeremy-creater/CourseScribe/blob/main/README.zh-CN.md
许可证：MIT，包内 Contents/Resources 里有 LICENSE、THIRD_PARTY_NOTICES.md 和 licenses 文件夹。


CourseScribe — record a course's sound or pick media files; get a transcript and a proofread version, all on your Mac
=====================================================================================================================

Usage (Apple silicon, macOS 14.2 or newer)
  1. Install and start Ollama (https://ollama.com), then run in Terminal: ollama pull gemma4:12b
     (for the study notes feature also: ollama pull qwen3.5:9b)
  2. Drag CourseScribe.app to Applications and open it.
  3. If macOS blocks the first start: System Settings -> Privacy & Security, find CourseScribe,
     click "Open Anyway". The app has no Apple developer signature.
  4. The first recording asks for permission to record system audio: allow it. A new version may ask again.
  5. Click the record button (or press Control+Option+R), watch your class, click stop. You can also
     drag audio or video files onto the window. The interface is in Chinese.
  6. The first task asks before it downloads the transcription model (about 3.1 GB). On a Mac the
     transcription runs on the processor, about 0.8x real time: a one-hour class takes about 48 minutes.

Results go to the CourseScribe folder in your Documents (in your home folder if Documents is synced by iCloud).
Files: settings, task state and logs are in ~/Library/Application Support/CourseScribe.
Uninstall: move the app to the Trash and delete that settings folder.

Full guide: https://github.com/Jeremy-creater/CourseScribe#readme
License: MIT; LICENSE, THIRD_PARTY_NOTICES.md and the licenses folder are in Contents/Resources.
