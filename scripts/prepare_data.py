"""Синтез вымышленных примеров Piper и приведение WAV к единому формату.

Запуск: .venv/Scripts/python.exe scripts/prepare_data.py
Повторная генерация намеренно требует --replace: исходные аудио эксперимента
нельзя незаметно заменить. Версии, SHA256, команды и результаты сохраняются.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import wave

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifest.json")
    parser.add_argument("--split", choices=("all", "development", "evaluation"), default="all")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--piper", type=Path, default=ROOT / ".runtime/tts/piper/piper.exe")
    parser.add_argument("--model", type=Path, default=ROOT / ".runtime/tts/ru_RU-irina-medium.onnx")
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    args = parser.parse_args()
    if not args.ffmpeg:
        parser.error("Не найден ffmpeg: установите или укажите --ffmpeg")
    manifest_path = args.manifest.resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    generation_path = manifest_path.parent / "generation.json"
    if generation_path.exists() and not args.replace:
        parser.error("Набор уже сгенерирован; используйте --replace только для объявленной новой версии")
    # Версии корпусов не перезаписывают промежуточные исходные WAV друг друга.
    raw_dir = ROOT / ".runtime/tts/generated" / manifest_path.parent.name
    raw_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "Piper 2023.11.14-2 portable Windows",
        "voice": "ru_RU-irina-medium",
        "model_sha256": digest(args.model),
        "piper_sha256": digest(args.piper),
        "manifest_sha256": digest(manifest_path),
        "format": "PCM signed 16-bit mono 16000 Hz",
        "validation": "Один синтетический голос. Эталон равен входу TTS. Независимая сверка прослушиванием человеком не выполнена. Клиническая разметка не утверждена врачом.",
        "files": [],
    }
    try:
        for item in data["items"]:
            if args.split != "all" and item["split"] != args.split:
                continue
            raw = raw_dir / (item["id"] + ".wav")
            audio, reference = ROOT / item["audio"], ROOT / item["reference"]
            audio.parent.mkdir(parents=True, exist_ok=True)
            reference.parent.mkdir(parents=True, exist_ok=True)
            command = [str(args.piper.resolve()), "--model", str(args.model.resolve()),
                       "--output_file", str(raw), "--length_scale", "1.0", "--sentence_silence", "0.2"]
            proc = subprocess.run(command, input=(item["text"] + "\n").encode("utf-8"),
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180,
                                  cwd=args.piper.parent, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            (raw_dir / (item["id"] + "-piper.log")).write_bytes(proc.stdout + proc.stderr)
            if proc.returncode:
                raise RuntimeError(f"Piper {item['id']} завершился с кодом {proc.returncode}")
            conversion = [args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw),
                          "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(audio)]
            converted = subprocess.run(conversion, capture_output=True, timeout=60,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if converted.returncode:
                raise RuntimeError(converted.stderr.decode("utf-8", errors="replace"))
            reference.write_text(item["text"], encoding="utf-8")
            with wave.open(str(audio), "rb") as stream:
                duration = stream.getnframes() / stream.getframerate()
                if not (stream.getnchannels() == 1 and stream.getframerate() == 16000
                        and stream.getsampwidth() == 2 and duration > 1):
                    raise RuntimeError(f"Некорректное аудио {audio}")
            report["files"].append({"id": item["id"], "split": item["split"], "audio": item["audio"],
                                    "duration_seconds": duration, "sha256": digest(audio),
                                    "reference_sha256": digest(reference), "piper_command": command,
                                    "ffmpeg_command": conversion})
            print(f"{item['id']}: {duration:.2f} s", flush=True)
        for control in data.get("controls", []):
            audio = ROOT / control["audio"]
            with wave.open(str(audio), "wb") as stream:
                stream.setnchannels(1)
                stream.setsampwidth(2)
                stream.setframerate(16000)
                stream.writeframes(b"\0\0" * int(16000 * control["duration_seconds"]))
            report["files"].append({"id": control["id"], "split": "control", "audio": control["audio"],
                                    "duration_seconds": control["duration_seconds"], "sha256": digest(audio)})
    except Exception as error:
        report["failure"] = repr(error)
        raise
    finally:
        generation_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
