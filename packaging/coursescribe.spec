# Build with: python -m PyInstaller packaging/coursescribe.spec --noconfirm
# Models are not included: the program uses the local cache and asks before downloading.
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

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

analysis = Analysis(
    [str(ROOT / "packaging" / "entry.py")],
    pathex=[str(ROOT / "src")],
    binaries=collect_dynamic_libs("ctranslate2") + collect_dynamic_libs("onnxruntime") + gpu_binaries,
    datas=collect_data_files("faster_whisper")
    + collect_data_files("soundcard")
    + [
        (str(ROOT / "src" / "coursescribe" / "ui" / "chevron.svg"), "coursescribe/ui"),
        # Beside notes_rules.py, where SHIPPED_RULES looks for it.
        (str(ROOT / "src" / "coursescribe" / "notes_rules.txt"), "coursescribe"),
    ],
    hiddenimports=["soundcard.mediafoundation", "_cffi_backend", "coursescribe.ui.app"],
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
)
COLLECT(exe, analysis.binaries, analysis.datas, name="CourseScribe")
