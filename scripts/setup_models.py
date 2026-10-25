"""Подготовить закреплённые локальные модели без установки Python-пакетов.

Обычный запуск: python scripts/setup_models.py
Для сравнения с Vosk: python scripts/setup_models.py --with-vosk
Системный прокси используется по умолчанию; --direct выбирает прямое соединение.
Загрузка не отправляет аудио или транскрипции. Рабочая обработка затем локальная.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import stat
import sys
import time
from urllib import error, parse, request
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASR_REVISION = "536b0662742c02347bc0e980a01041f333bce120"
QWEN_REVISION = "7dabda4d13d513e3e842b20f0d435c732f172cbe"
LLAMA_VERSION = "b11669"
QWEN_FILE = "qwen2.5-3b-instruct-q4_k_m.gguf"


@dataclass(frozen=True)
class Artifact:
    name: str
    url: str
    destination: str
    revision: str
    expected_sha256: str | None = None
    expected_size: int | None = None
    expected_git_blob: str | None = None
    license: str = ""


ARTIFACTS = [
    Artifact("Qwen2.5-3B-Instruct Q4_K_M",
             f"https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/{QWEN_REVISION}/{QWEN_FILE}",
             f"models/{QWEN_FILE}", QWEN_REVISION,
             expected_sha256="626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d",
             expected_size=2104932768, license="Qwen Research License"),
    Artifact("llama.cpp Windows CPU x64",
             f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_VERSION}/llama-{LLAMA_VERSION}-bin-win-cpu-x64.zip",
             f"downloads/llama-{LLAMA_VERSION}-bin-win-cpu-x64.zip", LLAMA_VERSION,
             expected_sha256="37b3efcf962847ca869745fd4ccf7f5e7e8dacfeff9fa3523b4ca8cdc66d632b",
             license="MIT"),
]

# Идентификаторы Git/LFS проверены через официальный API закреплённой ревизии:
# https://huggingface.co/api/models/Systran/faster-whisper-small/tree/536b0662742c02347bc0e980a01041f333bce120?expand=true
for filename, size, blob, sha in [
    ("config.json", 2370, "e5047537059bd8f182d9ca64c470201585015187", None),
    ("model.bin", 483546902, None,
     "3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671"),
    ("tokenizer.json", 2203239, "7818adb6de9fa3064d3ff81226fdd675be1f6344", None),
    ("vocabulary.txt", 459861, "c9074644d9d1205686f16d411564729461324b75", None),
]:
    ARTIFACTS.append(Artifact(
        f"faster-whisper-small/{filename}",
        f"https://huggingface.co/Systran/faster-whisper-small/resolve/{ASR_REVISION}/{filename}",
        f"models/faster-whisper-small/{filename}", ASR_REVISION,
        expected_sha256=sha, expected_size=size, expected_git_blob=blob, license="MIT"))

VOSK = Artifact("Vosk small Russian 0.22",
                "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip",
                "downloads/vosk-model-small-ru-0.22.zip", "0.22", license="Apache-2.0")

# Сохраняем уведомления рядом с компонентами; модель не входит в Git.
LICENSES = [
    Artifact("Qwen3B LICENSE",
             f"https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/{QWEN_REVISION}/LICENSE",
             "licenses/Qwen3B-LICENSE.txt", QWEN_REVISION, license="Qwen Research License"),
    Artifact("llama.cpp LICENSE",
             f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{LLAMA_VERSION}/LICENSE",
             "licenses/llama.cpp-LICENSE.txt", LLAMA_VERSION, license="MIT"),
    Artifact("faster-whisper model README",
             f"https://huggingface.co/Systran/faster-whisper-small/resolve/{ASR_REVISION}/README.md",
             "licenses/faster-whisper-small-README.md", ASR_REVISION, license="MIT"),
]


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, artifact: Artifact) -> str:
    if artifact.expected_size is not None and path.stat().st_size != artifact.expected_size:
        raise ValueError(f"{artifact.name}: размер не совпадает с закреплённым источником")
    digest = checksum(path)
    if artifact.expected_sha256 and digest != artifact.expected_sha256:
        raise ValueError(f"{artifact.name}: SHA256 не совпадает с опубликованным")
    if artifact.expected_git_blob:
        data = path.read_bytes()
        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if blob != artifact.expected_git_blob:
            raise ValueError(f"{artifact.name}: Git blob не совпадает с закреплённой ревизией")
    return digest


def download(artifact: Artifact, runtime: Path, direct: bool = False) -> dict:
    destination = runtime / artifact.destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        try:
            digest = verify(destination, artifact)
            print(f"Готово: {artifact.name} ({destination.stat().st_size:,} байт)", flush=True)
            return {**asdict(artifact), "actual_sha256": digest,
                    "actual_size": destination.stat().st_size, "downloaded": False}
        except ValueError:
            # Повреждённый установленный файл сохраняем для диагностики.
            destination.replace(destination.with_name(destination.name + ".invalid"))
    partial = destination.with_name(destination.name + ".part")
    opener = request.build_opener(request.ProxyHandler({}) if direct else request.ProxyHandler())
    for attempt in range(5):
        offset = partial.stat().st_size if partial.is_file() else 0
        url = artifact.url
        if attempt:
            sep = "&" if "?" in url else "?"
            url += f"{sep}download=true&x={time.time_ns()}"
        headers = {"User-Agent": "medical-audio-mvp-setup/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            print(f"Загрузка: {artifact.name}; продолжение с {offset:,} байт", flush=True)
            with opener.open(request.Request(url, headers=headers), timeout=90) as response:
                status = response.status
                if offset and status == 206:
                    expected_range = f"bytes {offset}-"
                    if not response.headers.get("Content-Range", "").startswith(expected_range):
                        raise ValueError("Сервер вернул другой диапазон файла")
                    mode = "ab"
                else:
                    # Сервер проигнорировал Range: начинаем файл целиком, без склейки.
                    mode = "wb"
                total = response.headers.get("Content-Length")
                expected_bytes = int(total) if total is not None else None
                written = 0
                next_progress = 256 * 1024 * 1024
                with partial.open(mode) as target:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        target.write(chunk)
                        written += len(chunk)
                        if written >= next_progress:
                            print(f"Получено: {artifact.name}: +{written // (1024 * 1024)} MiB", flush=True)
                            next_progress += 256 * 1024 * 1024
                if expected_bytes is not None and written != expected_bytes:
                    raise OSError("Соединение закрыто до окончания файла")
            digest = verify(partial, artifact)
            partial.replace(destination)
            print(f"Проверено: {artifact.name}; SHA256 {digest}", flush=True)
            return {**asdict(artifact), "actual_sha256": digest,
                    "actual_size": destination.stat().st_size, "downloaded": True}
        except error.HTTPError as exc:
            if exc.code == 416 and partial.is_file():
                try:
                    digest = verify(partial, artifact)
                    partial.replace(destination)
                    return {**asdict(artifact), "actual_sha256": digest,
                            "actual_size": destination.stat().st_size, "downloaded": True}
                except ValueError:
                    partial.unlink()
            if attempt == 4:
                raise RuntimeError(f"Не загружен {artifact.name}: HTTP {exc.code}") from exc
        except (OSError, ValueError) as exc:
            if isinstance(exc, ValueError) and partial.is_file():
                partial.unlink()
            if attempt == 4:
                raise RuntimeError(f"Не загружен {artifact.name}: {exc}") from exc
        time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Не загружен {artifact.name}")


def extract_zip(archive: Path, target: Path) -> None:
    """Каждый итоговый путь остаётся внутри указанного runtime-каталога."""
    resolved_target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for entry in source.infolist():
            path = (target / entry.filename).resolve()
            if not path.is_relative_to(resolved_target) or stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError("Недопустимый путь или ссылка в архиве runtime")
        source.extractall(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=PROJECT_ROOT / ".runtime")
    parser.add_argument("--with-vosk", action="store_true", help="Модель для отдельного сравнения ASR")
    parser.add_argument("--direct", action="store_true", help="Не использовать системный прокси")
    parser.add_argument("--plan", action="store_true", help="Только показать закреплённые источники")
    args = parser.parse_args()
    runtime = args.runtime.resolve()
    artifacts = ARTIFACTS + LICENSES + ([VOSK] if args.with_vosk else [])
    if args.plan:
        print(json.dumps([asdict(item) for item in artifacts], ensure_ascii=False, indent=2))
        return 0
    runtime.mkdir(parents=True, exist_ok=True)
    results = []
    failures = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = [(item, pool.submit(download, item, runtime, args.direct)) for item in artifacts]
        for item, future in pending:
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append({"name": item.name, "error": str(exc)})
                print(f"Ошибка: {exc}", file=sys.stderr, flush=True)
    archive = runtime / f"downloads/llama-{LLAMA_VERSION}-bin-win-cpu-x64.zip"
    if any(item["name"] == "llama.cpp Windows CPU x64" for item in results):
        extract_zip(archive, runtime / "llama")
    if args.with_vosk and any(item["name"] == VOSK.name for item in results):
        extract_zip(runtime / VOSK.destination, runtime / "models")
    manifest = {
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": str(runtime), "artifacts": results, "failures": failures,
        "verification": "SHA256 против опубликованного значения, где оно имеется; Git blob для малых ASR-файлов. Для Vosk публичная контрольная сумма не установлена: записан локально вычисленный SHA256.",
    }
    temporary = runtime / "manifest.json.part"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(runtime / "manifest.json")
    print(f"Манифест: {runtime / 'manifest.json'}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
