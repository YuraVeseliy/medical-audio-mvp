"""Скопировать только проверенные компоненты обработки в portable-комплект.

Модели и лицензии выбираются из manifest*.json, а не по фиксированному имени
Qwen. Файлы llama.cpp сверяются с закреплённым архивом. TTS-голос, аналоги,
Vosk для отдельного эксперимента и архивы загрузок в результат не копируются.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def file_hash(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def checked_relative(value: str) -> Path:
    normalized = value.replace("\\", "/")
    relative = Path(normalized)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Небезопасный путь в манифесте: {value}")
    return relative


def package_runtime(source: Path, destination: Path, verify_only: bool = False) -> dict:
    source, destination = source.resolve(), destination.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Исходная и целевая runtime-папки должны быть разными и не вложенными.")
    if destination.exists() and not verify_only:
        raise FileExistsError(f"Целевая runtime-папка уже существует: {destination}")
    manifests = sorted(source.glob("manifest*.json"))
    if not manifests:
        raise FileNotFoundError("В runtime не найден manifest*.json. Сначала подготовьте модели.")
    selected = {}
    llama_archives = []
    for record in manifests:
        data = json.loads(record.read_text(encoding="utf-8-sig"))
        if data.get("failures"):
            raise ValueError(f"В {record.name} отмечены неуспешные загрузки.")
        for item in data.get("artifacts", []):
            relative = checked_relative(item["destination"])
            # Первоначальная 1.5B остаётся только доказательством эксперимента A.
            # Рабочий комплект использует один выбранный GGUF.
            if relative.suffix.lower() == ".gguf" and relative.name != "qwen2.5-3b-instruct-q4_k_m.gguf":
                continue
            if relative.as_posix() == "licenses/Qwen-LICENSE.txt":
                continue
            if relative.parts[0] in {"models", "licenses"}:
                previous = selected.get(relative.as_posix())
                if previous and previous.get("actual_sha256") != item.get("actual_sha256"):
                    raise ValueError(f"Манифесты расходятся для {relative}")
                selected[relative.as_posix()] = item
            elif "llama" in item.get("name", "").lower() and relative.suffix.lower() == ".zip":
                llama_archives.append(item)
    if not any(path.endswith(".gguf") for path in selected):
        raise ValueError("Манифест не содержит GGUF-модель.")
    if "models/faster-whisper-small/model.bin" not in selected:
        raise ValueError("Манифест не содержит модель faster-whisper-small.")
    if not llama_archives:
        raise ValueError("Манифест не содержит закреплённый архив llama.cpp.")

    entries = []
    copies = []
    for relative_text, item in sorted(selected.items()):
        relative = checked_relative(relative_text)
        original = source / relative
        digest = file_hash(original)
        expected = item.get("expected_sha256") or item.get("actual_sha256")
        if not expected or digest != expected:
            raise ValueError(f"Контрольная сумма runtime не совпадает: {relative}")
        entries.append({"path": relative.as_posix(), "size": original.stat().st_size,
                        "sha256": digest, "source_manifest_artifact": item.get("name")})
        copies.append((original, relative))

    # При нескольких версиях llama.cpp не выбираем молча случайную.
    if len({item["destination"] for item in llama_archives}) != 1:
        raise ValueError("Обнаружено несколько архивов llama.cpp. Оставьте один в актуальном манифесте.")
    item = llama_archives[0]
    archive_path = source / checked_relative(item["destination"])
    if file_hash(archive_path) != (item.get("expected_sha256") or item.get("actual_sha256")):
        raise ValueError("Контрольная сумма архива llama.cpp не совпадает.")
    found_server = False
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            basename = Path(member.filename).name
            if member.is_dir() or not (basename == "llama-server.exe" or
                                       basename.lower().endswith(".dll") or
                                       basename.startswith("LICENSE")):
                continue
            relative = Path("llama") / basename
            original = source / relative
            expected = hashlib.sha256(archive.read(member)).hexdigest()
            if file_hash(original) != expected:
                raise ValueError(f"Извлечённый llama.cpp отличается от архива: {relative}")
            found_server |= basename == "llama-server.exe"
            entries.append({"path": relative.as_posix(), "size": original.stat().st_size,
                            "sha256": expected, "source_archive": item["destination"]})
            copies.append((original, relative))
    if not found_server:
        raise ValueError("В закреплённом архиве не найден llama-server.exe.")
    summary = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
               "source": str(source), "destination": str(destination),
               "verification_only": verify_only, "files": entries,
               "excluded": ["downloads", "tts", "analogs", "vosk comparison model", "logs"]}
    if not verify_only:
        destination.mkdir(parents=True)
        for original, relative in copies:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
        for record in manifests:
            shutil.copy2(record, destination / record.name)
        (destination / "portable-manifest.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / ".runtime")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = package_runtime(args.source, args.destination, args.verify_only)
    print(json.dumps({"files": len(result["files"]),
                      "bytes": sum(item["size"] for item in result["files"]),
                      "destination": result["destination"],
                      "verification_only": args.verify_only}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
