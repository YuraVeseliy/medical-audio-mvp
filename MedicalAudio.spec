# -*- mode: python ; coding: utf-8 -*-
"""Сборка Windows x64 через scripts/build_portable.ps1.

GGUF, ASR и llama.cpp лежат рядом с EXE в .runtime, а не внутри one-file EXE.
Сборка на CPU не требует пакетов CUDA.
"""
from pathlib import Path
from PyInstaller.utils.hooks import collect_all


root = Path(SPECPATH)
datas = []
binaries = []
hiddenimports = []
for package in ("faster_whisper", "ctranslate2", "av", "tokenizers",
                "onnxruntime", "sounddevice", "soundfile", "_sounddevice_data"):
    package_data, package_binaries, package_hidden = collect_all(package)
    datas += package_data
    binaries += package_binaries
    hiddenimports += package_hidden

a = Analysis(
    [str(root / "run_app.pyw")],
    pathex=[str(root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[], hooksconfig={}, runtime_hooks=[],
    excludes=["torch", "tensorflow", "pytest", "IPython", "notebook"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True,
    name="MedicalAudio", debug=False, bootloader_ignore_signals=False,
    strip=False, upx=False, console=False, disable_windowed_traceback=False,
    argv_emulation=False, target_arch=None, codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False,
               upx_exclude=[], name="MedicalAudio")
