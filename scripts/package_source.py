"""Создать архив исходников и проверяемый перечень его содержимого.

Модели, виртуальное окружение, история Git и пользовательские сеансы не входят.
Запуск: python scripts/package_source.py [--output путь.zip] [--list-only]
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {
    ".git", ".runtime", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "tmp", "dist", "build", "user_sessions", "node_modules",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".tmp", ".key", ".pem"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_files(output: Path) -> list[Path]:
    result = []
    for item in ROOT.rglob("*"):
        relative = item.relative_to(ROOT)
        if any(part in EXCLUDED_DIRS for part in relative.parts):
            continue
        if item.is_symlink():
            raise ValueError(f"Символьная ссылка не упаковывается: {relative}")
        if not item.is_file() or item.resolve() == output.resolve():
            continue
        if item.suffix.lower() in EXCLUDED_SUFFIXES or item.name.startswith(".env"):
            continue
        if item.name.endswith(".log.tmp"):
            continue
        result.append(item)
    return sorted(result, key=lambda item: item.relative_to(ROOT).as_posix())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT.parents[1] / "output/medical_audio_mvp_source.zip")
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    files = source_files(output)
    if args.list_only:
        for item in files:
            print(item.relative_to(ROOT).as_posix())
        print(f"Всего: {len(files)} файлов")
        return 0
    if output.exists():
        raise FileExistsError(f"Архив уже существует: {output}. Укажите новое имя --output.")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    entries = []
    for item in files:
        data = item.read_bytes()
        entries.append({"path": item.relative_to(ROOT).as_posix(),
                        "size": len(data), "sha256": sha256(data)})
    manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                "project": "medical_audio_mvp", "git_head_at_packaging": commit,
                "note": "Архив отражает фактические файлы, включая ещё не закоммиченные; проверяйте SHA256.",
                "excluded_directories": sorted(EXCLUDED_DIRS), "files": entries}
    # Состав и содержимое контролируются в одном проходе: изменение во время
    # упаковки обнаруживается, вместо создания несогласованного архива.
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as archive:
        for item, entry in zip(files, entries):
            data = item.read_bytes()
            if sha256(data) != entry["sha256"]:
                raise RuntimeError(f"Файл изменился во время упаковки: {entry['path']}")
            archive.writestr("medical_audio_mvp/" + entry["path"], data)
        # Только записи происхождения, без бинарных моделей и TTS-голоса.
        for record in sorted((ROOT / ".runtime").glob("manifest*.json")):
            archive.writestr("medical_audio_mvp/_package/runtime-evidence/" + record.name,
                             record.read_bytes())
        archive.writestr("medical_audio_mvp/_package/source-manifest.json",
                         json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
    with output.open("rb") as archive_file:
        digest = hashlib.file_digest(archive_file, "sha256").hexdigest()
    summary = {"archive": str(output), "sha256": digest, "size": output.stat().st_size,
               "files": len(entries), "git_head_at_packaging": commit,
               "created_at_utc": manifest["created_at_utc"]}
    output.with_suffix(".zip.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
