# Build with: python -m PyInstaller packaging/coursescribe.spec --noconfirm
# Models are not included: the program uses the local cache and asks before downloading.
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = Path(SPECPATH).parent
SITE = Path(sys.prefix) / "Lib" / "site-packages"

# Only what the engine loads. The rest of the CUDA packages would add gigabytes.
GPU_LIBRARIES = {
    "cublas": ["cublas64_12.dll", "cublasLt64_12.dll"],
    "cudnn": ["cudnn_ops64_9.dll", "cudnn_cnn64_9.dll"],
}
gpu_binaries = []
if sys.platform == "win32":
    for package, names in GPU_LIBRARIES.items():
        folder = SITE / "nvidia" / package / "bin"
        for name in names:
            if (folder / name).exists():
                gpu_binaries.append((str(folder / name), f"nvidia/{package}/bin"))

mac_binaries, mac_datas, mac_hidden = [], [], []
if sys.platform == "darwin":
    # The Core Audio tap library ships its own compiled helper; PyObjC loads its frameworks by name.
    mac_binaries = collect_dynamic_libs("catap")
    mac_datas = collect_data_files("catap")
    mac_hidden = collect_submodules("catap") + ["objc", "Foundation", "CoreAudio", "AppKit"]

analysis = Analysis(
    [str(ROOT / "packaging" / "entry.py")],
    pathex=[str(ROOT / "src")],
    binaries=collect_dynamic_libs("ctranslate2") + collect_dynamic_libs("onnxruntime") + gpu_binaries + mac_binaries,
    datas=collect_data_files("faster_whisper")
    + (collect_data_files("soundcard") if sys.platform == "win32" else [])
    + mac_datas
    + [
        (str(ROOT / "src" / "coursescribe" / "ui" / "chevron.svg"), "coursescribe/ui"),
        # Beside notes_rules.py, where SHIPPED_RULES looks for it.
        (str(ROOT / "src" / "coursescribe" / "notes_rules.txt"), "coursescribe"),
    ],
    hiddenimports=["coursescribe.ui.app"]
    + (["soundcard.mediafoundation", "_cffi_backend"] if sys.platform == "win32" else mac_hidden),
    excludes=["tkinter", "matplotlib", "pytest", "IPython", "torch"],
)
pyz = PYZ(analysis.pure)
exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="CourseScribe",
    console=False,
    icon=str(ROOT / "packaging" / "icons" / "CourseScribe.ico"),
)
coll = COLLECT(exe, analysis.binaries, analysis.datas, name="CourseScribe")
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="CourseScribe.app",
        icon=str(ROOT / "packaging" / "icons" / "CourseScribe.icns"),
        bundle_identifier="app.coursescribe.CourseScribe",
        info_plist={
            "CFBundleName": "CourseScribe",
            "NSHighResolutionCapable": True,
            # Shown in the system's permission window when recording the Mac's sound.
            "NSAudioCaptureUsageDescription": "CourseScribe 需要录下电脑播放的声音，用来把网课转成文字。录音只保存在这台电脑上。",
        },
    )
