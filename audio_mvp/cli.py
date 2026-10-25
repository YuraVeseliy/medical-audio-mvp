"""Командный запуск той же обработки, что используется интерфейсом."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import sys

from .model import AppError
from .providers import ASRRecognizer, MODEL_BY_ID, default_model_id
from .session import Session
from .storage import export_text


def main(argv=None):
    parser = argparse.ArgumentParser(description="Локальная обработка вымышленного медицинского диалога")
    parser.add_argument("audio", type=Path)
    project_root = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
    parser.add_argument("--runtime", type=Path, default=project_root / ".runtime")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", choices=list(MODEL_BY_ID), default=default_model_id())
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    recognizer = ASRRecognizer(args.model, args.runtime, args.threads)
    try:
        session = Session(recognizer)
        started = time.perf_counter()
        document = session.process_file(args.audio)
        elapsed = time.perf_counter() - started
        args.out.mkdir(parents=True, exist_ok=True)
        session.save(args.out / "session.json")
        export_text(document, args.out / "result.txt")
        (args.out / "run.json").write_text(json.dumps(
            {"wall_seconds_including_first_model_load": elapsed, "document": asdict(document)},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Готово: {args.out}; обработка {elapsed:.2f} с")
        return 0
    except AppError as exc:
        print(str(exc))
        return 1
    finally:
        recognizer.close()


if __name__ == "__main__":
    raise SystemExit(main())
