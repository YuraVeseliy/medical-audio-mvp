"""Закреплённая исследовательская модель-кандидат; не отправляет учебные данные."""
from datetime import datetime, timezone
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import shutil
import time
from urllib import request
from setup_models import Artifact, download

REVISION = "7dabda4d13d513e3e842b20f0d435c732f172cbe"
BASE = f"https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/{REVISION}"
ARTIFACTS = [
    Artifact("Qwen2.5-3B-Instruct Q4_K_M", BASE + "/qwen2.5-3b-instruct-q4_k_m.gguf",
             "models/qwen2.5-3b-instruct-q4_k_m.gguf", REVISION,
             expected_sha256="626b4a6678b86442240e33df819e00132d3ba7dddfe1cdc4fbb18e0a9615c62d",
             expected_size=2104932768, license="Qwen Research License"),
    Artifact("Qwen3B LICENSE", BASE + "/LICENSE", "licenses/Qwen3B-LICENSE.txt", REVISION,
             expected_size=7388, expected_git_blob="ed317377ac0abff39f17eec693ce664f4b8152af",
             license="Qwen Research License"),
    Artifact("Qwen3B model card", BASE + "/README.md", "licenses/Qwen3B-README.md", REVISION,
             expected_size=4867, expected_git_blob="334d54421ca2e346f3d2a1265236d6da0992a436",
             license="Qwen Research License"),
]

def parallel_model(root):
    """Диапазоны сверяются сервером; итоговый SHA проверяет общий загрузчик."""
    item = ARTIFACTS[0]
    target = root / item.destination
    if target.is_file():
        return
    partial = target.with_name(target.name + ".part")
    partial.parent.mkdir(parents=True, exist_ok=True)
    prefix = partial.stat().st_size if partial.exists() else 0
    folder = root / "downloads" / "qwen3b-parts"
    folder.mkdir(parents=True, exist_ok=True)
    size = item.expected_size
    step = 32 * 1024 * 1024
    spans = []
    for start in range(prefix, size, step):
        end = min(start + step, size)-1
        existing = folder / f"{start}-{end}"
        if existing.is_file() and existing.stat().st_size == end-start+1:
            spans.append((start,end))
        else:
            spans.extend((part,min(part+2*1024*1024,end+1)-1)
                         for part in range(start,end+1,2*1024*1024))
    def fetch(span):
        start, end = span
        chunk = folder / f"{start}-{end}"
        if chunk.is_file() and chunk.stat().st_size == end-start+1:
            return chunk
        for attempt in range(4):
            try:
                opener = request.build_opener(request.ProxyHandler({}))
                req = request.Request(item.url + f"?download=true&range={start}-{end}",
                                      headers={"Range": f"bytes={start}-{end}"})
                with opener.open(req, timeout=90) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise ValueError("Сервер не подтвердил диапазон")
                    with chunk.with_suffix(".tmp").open("wb") as out:
                        shutil.copyfileobj(response, out, 1024*1024)
                if chunk.with_suffix(".tmp").stat().st_size != end-start+1:
                    raise ValueError("Неполный диапазон")
                chunk.with_suffix(".tmp").replace(chunk)
                print(f"Диапазон {start//1048576}–{end//1048576} MiB получен", flush=True)
                return chunk
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2)
    with ThreadPoolExecutor(max_workers=16) as pool:
        chunks = list(pool.map(fetch, spans))
    with partial.open("ab") as out:
        for chunk in chunks:
            with chunk.open("rb") as source:
                shutil.copyfileobj(source, out, 1024*1024)

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / ".runtime"
    parallel_model(root)
    records = [download(item, root, direct=True) for item in ARTIFACTS]
    manifest = {"prepared_at_utc": datetime.now(timezone.utc).isoformat(),
                "artifacts": records, "failures": [],
                "purpose": "Некоммерческое учебное исследование. Условия Qwen Research License приложены; для коммерческого применения требуется отдельное рассмотрение лицензии."}
    (root / "manifest-3b.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
