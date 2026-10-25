"""Одна команда быстрых проверок; модели и микрофон не требуются."""
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import platform
import sys
import unittest
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    os.chdir(ROOT)
    from audio_mvp.provenance import source_version
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'reports/asr_checks')
    args = parser.parse_args()
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    output = stream.getvalue()
    print(output)
    report = {"time_utc": datetime.now(timezone.utc).isoformat(), **source_version(ROOT), "python": sys.version,
              "platform": platform.platform(), "executed_by": "automatic unittest runner",
              "scope": "public transcription operations, file IO, full audio coverage, WER example and recording stop; external ASR/device doubles, not model accuracy",
              "run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
              "skipped": len(result.skipped), "successful": result.wasSuccessful()}
    target = args.out
    target.mkdir(parents=True, exist_ok=True)
    (target / "tests.log").write_text(output, encoding="utf-8")
    (target / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
