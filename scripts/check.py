"""Одна команда быстрых проверок; модели и микрофон не требуются."""
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    os.chdir(ROOT)
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    output = stream.getvalue()
    print(output)
    files = sorted((ROOT / "audio_mvp").glob("*.py")) + sorted((ROOT / "tests").glob("*.py"))
    files += [ROOT / "run_app.pyw", Path(__file__)]
    digest = hashlib.sha256()
    for item in files:
        digest.update(item.relative_to(ROOT).as_posix().encode())
        digest.update(item.read_bytes())
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "git unavailable; see source_sha256"
    report = {"time_utc": datetime.now(timezone.utc).isoformat(), "commit": commit,
              "source_sha256": digest.hexdigest(), "python": sys.version,
              "hashed_files": [item.relative_to(ROOT).as_posix() for item in files],
              "platform": platform.platform(), "executed_by": "automatic unittest runner",
              "scope": "public Session operations with external ASR/LLM test doubles; not model accuracy",
              "run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
              "skipped": len(result.skipped), "successful": result.wasSuccessful()}
    target = ROOT / "reports/checks"
    target.mkdir(parents=True, exist_ok=True)
    (target / "tests.log").write_text(output, encoding="utf-8")
    (target / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
