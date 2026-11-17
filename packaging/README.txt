CourseScribe — 录下网课声音或选音视频文件，得到文字稿和校对稿，全程在本机
==========================================================================

使用
  1. 先安装并启动 Ollama（https://ollama.com），然后在命令行运行：ollama pull gemma4:12b
     （要用“整理笔记”再运行：ollama pull qwen3.5:9b）
  2. 双击 CourseScribe.exe。_internal 文件夹要和它放在一起，不能单独拿走 exe。
  3. 第一次运行 Windows 可能提示“Windows 已保护你的电脑”：点“更多信息”→“仍要运行”。
     这是因为程序没有付费的代码签名，不是病毒。
  4. 点“开始录音”（或按 Ctrl+Alt+R）照常看课，看完点“停止”。也可以把音视频文件拖进窗口。
  5. 第一个任务会问你是否下载转录模型（约 3.1 GB），同意后才下载。

结果默认在“文档”里的 CourseScribe 文件夹：.txt 是原始文字稿，.md 是校对稿。
设置（模型、保存文件夹、词表、整理规则、快捷键）在主窗口的“设置”里。

文件位置
  设置、任务状态、日志  %LOCALAPPDATA%\CourseScribe

更新：退出程序，把新版本解压覆盖旧文件夹，设置和任务保留。
卸载：删除这个文件夹和 %LOCALAPPDATA%\CourseScribe；结果文件夹要不要删由你决定。

完整说明（中文）https://github.com/Jeremy-creater/CourseScribe/blob/main/README.zh-CN.md
许可证：MIT（LICENSE），第三方组件见 THIRD_PARTY_NOTICES.md 和 licenses 文件夹。


CourseScribe — record a course's sound or pick media files; get a transcript and a proofread version, all on your computer
==========================================================================================================================

Usage
  1. Install and start Ollama (https://ollama.com), then run: ollama pull gemma4:12b
     (for the study notes feature also: ollama pull qwen3.5:9b)
  2. Double-click CourseScribe.exe. Keep the _internal folder next to it.
  3. On the first start Windows may say "Windows protected your PC": click "More info", then
     "Run anyway". The app has no paid code-signing certificate.
  4. Click the record button (or press Ctrl+Alt+R), watch your class, click stop. You can also
     drag audio or video files onto the window. The interface is in Chinese.
  5. The first task asks before it downloads the transcription model (about 3.1 GB).

Results go to the CourseScribe folder in your Documents by default: .txt is the raw transcript,
.md the proofread version. Settings are in the main window.

Files: settings, task state and logs are in %LOCALAPPDATA%\CourseScribe.
Update: quit the program and unzip the new version over this folder; settings and tasks are kept.
Uninstall: delete this folder and %LOCALAPPDATA%\CourseScribe; your results are yours to keep or delete.

Full guide: https://github.com/Jeremy-creater/CourseScribe#readme
License: MIT (LICENSE); third-party components: THIRD_PARTY_NOTICES.md and the licenses folder.
