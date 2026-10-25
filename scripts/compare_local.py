"""Воспроизводимые измерения; модельные эксперименты отделены от быстрых тестов.

Примеры:
  python scripts/compare_local.py inventory
  python scripts/compare_local.py asr --engine faster-whisper --model .runtime/whisper-small --out reports/asr-fw
  python scripts/compare_local.py asr --engine vosk --model .runtime/vosk-model-small-ru-0.22 --out reports/asr-vosk
  python scripts/compare_local.py extract --engine oms --endpoint http://127.0.0.1:8787 --out reports/extract-oms

Все входы вымышленные. Итоговая оценка не является клинической валидацией.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import re
import statistics
import sys
import time
from types import SimpleNamespace
import urllib.request
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROMPT = """Ты извлекаешь дословные цитаты из вымышленного диалога врача и пациента.
Верни только JSON с ключами noteText, sections, codingHints, followUpQuestions, warnings.
sections содержит два массива: complaints (нынешние жалобы, симптомы и их отрицания)
и anamnesis (начало и развитие состояния, предшествующие события, перенесённые состояния,
вмешательства и предпринятые действия). Каждый элемент массива — объект {\"quote\": \"дословная цитата\"}.
Одна цитата должна выражать один факт. Сохраняй отрицания, время и сторону тела.
Извлекай только ответы пациента. Не переписывай вопросы врача как факты.
Копируй цитаты из исходного текста буквально. Не ставь диагноз и ничего не додумывай.
Отсутствующие сведения не добавляй. Не придумывай обязательные поля или рекомендации.
noteText — пустая строка; codingHints, followUpQuestions, warnings — пустые массивы."""


def normalize(value: str) -> str:
    """Одинаково для эталона и ответа: регистр, ё/е, пунктуация, пробелы."""
    return " ".join(re.findall(r"[\w]+", value.casefold().replace("ё", "е"), re.UNICODE))


def words(value: str) -> list[str]:
    return normalize(value).split()


def wer_counts(reference: str, hypothesis: str) -> dict:
    """Левенштейн по словам; при равенстве приоритет S, D, I фиксирован."""
    ref, hyp = words(reference), words(hypothesis)
    previous = [(j, 0, 0, j) for j in range(len(hyp) + 1)]
    for i, left in enumerate(ref, 1):
        current = [(i, 0, i, 0)]
        for j, right in enumerate(hyp, 1):
            if left == right:
                current.append(previous[j - 1])
            else:
                cost, s, d, ins = previous[j - 1]
                substitution = (cost + 1, s + 1, d, ins)
                cost, s, d, ins = previous[j]
                deletion = (cost + 1, s, d + 1, ins)
                cost, s, d, ins = current[j - 1]
                insertion = (cost + 1, s, d, ins + 1)
                current.append(min((substitution, deletion, insertion), key=lambda x: x[0]))
        previous = current
    error, subs, deletes, inserts = previous[-1]
    return {"reference_words": len(ref), "hypothesis_words": len(hyp), "substitutions": subs,
            "deletions": deletes, "insertions": inserts, "errors": error,
            "wer": error / len(ref) if ref else None}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def environment() -> dict:
    packages = {}
    for package in ("faster-whisper", "ctranslate2", "vosk", "soundfile", "numpy", "openai", "buzz-captions"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "processor": platform.processor(), "logical_cpus": os.cpu_count(), "packages": packages}


def manifest(path: Path) -> dict:
    dataset = json.loads(path.read_text(encoding="utf-8-sig"))
    ids = [item["id"] for item in dataset["items"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Повторяющиеся ID в manifest")
    for item in dataset["items"]:
        for fact in item["expected"]:
            if fact["quote"] not in item["text"]:
                raise ValueError(f"{item['id']}: эталонная цитата отсутствует в исходном тексте")
            if fact["section"] not in {"complaints", "anamnesis"}:
                raise ValueError(f"Неизвестный раздел: {fact['section']}")
            if not all(normalize(needle) in normalize(fact["quote"]) for needle in fact["needles"]):
                raise ValueError(f"{item['id']}: ключевая фраза отсутствует в цитате")
    return dataset


def select_items(dataset: dict, split: str) -> list[dict]:
    if split == "control":
        return [dict(item, split="control", expected=[]) for item in dataset["controls"]]
    return [item for item in dataset["items"] if split == "all" or item["split"] == split]


def audio_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as stream:
        return stream.getnframes() / stream.getframerate()


def asr_adapter(engine: str, model: str, threads: int):
    if engine == "faster-whisper":
        from faster_whisper import WhisperModel
        recognizer = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=threads)

        def transcribe(path: Path):
            segments, info = recognizer.transcribe(str(path), language="ru", beam_size=5, vad_filter=True,
                                                  condition_on_previous_text=False)
            # Генератор обязательно исчерпан до остановки таймера.
            captured = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
            return " ".join(s["text"].strip() for s in captured).strip(), {
                "segments": captured, "language": info.language, "language_probability": info.language_probability}
        return transcribe
    if engine == "vosk":
        from vosk import Model, KaldiRecognizer, SetLogLevel
        SetLogLevel(-1)
        recognizer_model = Model(model)

        def transcribe(path: Path):
            results = []
            with wave.open(str(path), "rb") as stream:
                if stream.getnchannels() != 1 or stream.getsampwidth() != 2 or stream.getframerate() != 16000:
                    raise ValueError("Vosk эксперимент ожидает PCM16 mono 16000 Hz")
                recognizer = KaldiRecognizer(recognizer_model, 16000)
                while chunk := stream.readframes(4000):
                    if recognizer.AcceptWaveform(chunk):
                        results.append(json.loads(recognizer.Result()))
                results.append(json.loads(recognizer.FinalResult()))
            return " ".join(item.get("text", "") for item in results).strip(), {"recognizer_results": results}
        return transcribe
    raise ValueError(engine)


def post_json(url: str, body: dict, timeout: float = 240):
    request = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_json_text(raw):
    if isinstance(raw, dict):
        return raw
    text = str(raw).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    return json.loads(text)


def indexed_analog_prompt() -> str:
    """Та же задача и development-пример, в штатной JSON-обёртке OMS."""
    from audio_mvp.providers_indexed import PROMPT as indexed_prompt
    prompt = indexed_prompt.replace(
        '{"complaints":[],"anamnesis":[]}.',
        '{"noteText":"","sections":{"complaints":[],"anamnesis":[]},'
        '"codingHints":[],"followUpQuestions":[],"warnings":[]}.', 1)
    prompt = prompt.replace(
        'Ответ: {"complaints":[2],"anamnesis":[4,6,8]}.',
        'Ответ: {"noteText":"","sections":{"complaints":[2],"anamnesis":[4,6,8]},'
        '"codingHints":[],"followUpQuestions":[],"warnings":[]}.')
    return prompt + "\nПоля noteText, codingHints, followUpQuestions, warnings оставь пустыми. Номера помещай только в sections."


def numbered_source(text: str) -> str:
    from audio_mvp.providers_indexed import build_request
    return build_request(text)["messages"][1]["content"]


def restore_indexed_output(raw: dict, source_text: str) -> dict:
    """Общий технический декодер номеров, без медицинских правил/подсказок."""
    from audio_mvp.providers_indexed import parse_indexed_evidence, split_candidates
    try:
        parsed = parse_json_text(raw["raw_text"]) if "raw_text" in raw else raw
        sections = parsed.get("sections")
        if not isinstance(sections, dict):
            raise ValueError("Ответ indexed-аналога не содержит sections")
        candidates = split_candidates(source_text)
        evidence, warnings = parse_indexed_evidence(candidates, sections)
    except Exception as error:
        error.model_response = raw
        raise
    return {"evidence": [asdict(item) for item in evidence], "warnings": warnings,
            "raw_provider": raw, "candidates": [asdict(item) for item in candidates]}


def extraction_adapter(args, prompt: str):
    if args.engine == "oms":
        def extract(text: str):
            raw = post_json(args.endpoint.rstrip("/") + "/v1/encounters/scribe",
                            {"transcript": numbered_source(text) if args.indexed else text,
                             "language": "ru", "noteStyle": "custom", "customPrompt": prompt})
            return restore_indexed_output(raw, text) if args.indexed else raw
        return extract
    if args.engine == "buzz":
        if args.buzz_source:
            sys.path.insert(0, str(Path(args.buzz_source).resolve()))
        from buzz.plugins.ai_summary.plugin import AISummaryPlugin
        plugin = AISummaryPlugin()
        config = {"api_url": args.endpoint.rstrip("/"), "api_key": "local-only", "model": args.model,
                  "prompt": prompt, "save_to_notes": False, "save_to_file": False}
        context = SimpleNamespace(config=config, log=logging.getLogger("buzz-experiment"))

        def extract(text: str):
            answer = plugin._summarize(config, numbered_source(text) if args.indexed else text, context)
            if answer is None:
                raise RuntimeError("Штатный AI Summary Buzz вернул пустой ответ/ошибку; см. журнал")
            raw = {"component": "AISummaryPlugin._summarize", "raw_text": answer}
            return restore_indexed_output(raw, text) if args.indexed else raw
        return extract
    if args.engine in {"our", "our-indexed"}:
        # Стабильный вход эксперимента — конкретный публичный адаптер извлечения.
        if args.engine == "our-indexed":
            from audio_mvp.providers_indexed import IndexedExtractor
            extractor = IndexedExtractor(base_url=args.endpoint, model=args.model)
        else:
            from audio_mvp.providers import LocalExtractor
            extractor = LocalExtractor(base_url=args.endpoint, model=args.model)

        def extract(text: str):
            try:
                evidence, warnings = extractor.extract(text)
            except Exception as error:
                # Не теряем ответ, который производственный адаптер отклонил.
                error.model_response = extractor.last_response
                raise
            return {"evidence": [asdict(item) if is_dataclass(item) else item for item in evidence],
                    "warnings": warnings, "model_response": extractor.last_response,
                    "prompt_tokens": extractor.last_prompt_tokens}
        return extract
    raise ValueError(args.engine)


def pipeline_adapter(args):
    """Настоящий публичный сценарий приложения, а не сумма отдельных медиан."""
    from audio_mvp.providers import FasterWhisperRecognizer, LocalExtractor
    from audio_mvp.providers_indexed import IndexedExtractor
    from audio_mvp.session import Session
    extractor_type = IndexedExtractor if args.engine == "our-indexed" else LocalExtractor
    extractor = extractor_type(base_url=args.endpoint, model=args.model)
    session = Session(FasterWhisperRecognizer(args.asr_model), extractor)

    def process(path: Path):
        session.new(force=True)
        try:
            document = session.process_file(path)
        except Exception as error:
            error.model_response = extractor.last_response
            raise
        return {"document": asdict(document), "model_response": extractor.last_response,
                "prompt_tokens": extractor.last_prompt_tokens,
                "evidence": [asdict(item) for item in document.evidence]}
    return process


def extracted_items(raw: dict) -> list[dict]:
    if "raw_text" in raw:
        raw = parse_json_text(raw["raw_text"])
    if "parsed" in raw:
        raw = raw["parsed"]
    if "evidence" in raw:
        return [{"section": item.get("section", ""), "quote": str(item.get("quote", ""))}
                for item in raw["evidence"]]
    sections = raw.get("sections")
    if not isinstance(sections, dict):
        raise ValueError("Ответ не содержит sections или evidence")
    items = []
    for section, entries in sections.items():
        if isinstance(entries, str):
            entries = [line for line in entries.splitlines() if line.strip()]
        elif not isinstance(entries, list):
            entries = [entries]
        for entry in entries:
            quote = entry.get("quote", entry.get("text", "")) if isinstance(entry, dict) else str(entry)
            if str(quote).strip():
                items.append({"section": section, "quote": str(quote).strip()})
    return items


def score_extraction(expected: list[dict], predictions: list[dict], source_text: str) -> dict:
    """Прозрачный технический подсчёт; человеческий разбор смысла нужен отдельно.

    Один прогноз ↔ максимум один эталонный факт. Для совпадения нужны верный
    раздел, точное нормализованное совпадение с эталонной цитатой и источник.
    Простое наличие ключевых слов ошибочно засчитало бы весь диалог как факт.
    Дубликаты, неверный раздел и неподтверждённые цитаты остаются ошибками.
    """
    remaining = set(range(len(expected)))
    matched, unmatched = [], []
    source = normalize(source_text)
    for index, prediction in enumerate(predictions):
        quote = normalize(prediction["quote"])
        grounded = bool(quote) and quote in source
        match = None
        if grounded:
            for candidate in sorted(remaining):
                fact = expected[candidate]
                if prediction["section"] == fact["section"] and quote == normalize(fact["quote"]):
                    match = candidate
                    break
        entry = {"prediction_index": index, **prediction, "grounded": grounded}
        if match is None:
            unmatched.append(entry)
        else:
            remaining.remove(match)
            matched.append({**entry, "expected_index": match})
    tp, count, total = len(matched), len(predictions), len(expected)
    return {"true_positive": tp, "predicted": count, "expected": total,
            "false_positive": count - tp, "false_negative": total - tp,
            "precision": tp / count if count else None, "recall": tp / total if total else None,
            "ungrounded_predictions": sum(not item["grounded"] for item in unmatched),
            "matched": matched, "unmatched": unmatched,
            "missed": [{"expected_index": i, **expected[i]} for i in sorted(remaining)]}


def input_text(item: dict, args) -> str:
    if args.transcript_dir:
        return (Path(args.transcript_dir) / f"{item['id']}.txt").read_text(encoding="utf-8-sig")
    return item["text"]


def summarize(runs: list[dict], operation: str) -> dict:
    successful = [run for run in runs if run["status"] == "completed"]
    summary = {"attempted": len(runs), "completed": len(successful), "failed": len(runs) - len(successful)}
    elapsed = [run["seconds"] for run in successful]
    if elapsed:
        summary["seconds"] = {"median": statistics.median(elapsed), "min": min(elapsed), "max": max(elapsed)}
    if operation == "asr":
        count = Counter()
        for run in successful:
            for key in ("errors", "reference_words", "substitutions", "deletions", "insertions"):
                count[key] += run["metrics"][key]
        summary["word_counts"] = dict(count)
        summary["corpus_wer"] = count["errors"] / count["reference_words"] if count["reference_words"] else None
    else:
        tp = sum(run["metrics"]["true_positive"] for run in successful)
        predicted = sum(run["metrics"]["predicted"] for run in successful)
        expected = sum(run["metrics"]["expected"] for run in successful)
        summary.update({"true_positive": tp, "predicted": predicted, "expected_in_completed_runs": expected,
                        "precision": tp / predicted if predicted else None, "recall": tp / expected if expected else None})
    summary["denominator_note"] = "Качество суммируется по завершённым попыткам; ошибки запуска показаны отдельно. Повторы не являются независимыми сценариями."
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("inventory", "asr", "extract", "pipeline"))
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifest.json")
    parser.add_argument("--split", choices=("development", "evaluation", "all", "control"), default="evaluation")
    parser.add_argument("--engine", default="faster-whisper")
    parser.add_argument("--indexed", action="store_true", help="Общая задача выбора номеров для штатных OMS/Buzz")
    parser.add_argument("--model", default="medical-local")
    parser.add_argument("--asr-model", default=str(ROOT / ".runtime/models/faster-whisper-small"))
    parser.add_argument("--endpoint", default="http://127.0.0.1:8787")
    parser.add_argument("--model-server", default="", help="Локальный llama-server для фиксации фактических /props (включая модели аналогов)")
    parser.add_argument("--buzz-source", default="")
    parser.add_argument("--transcript-dir", default="")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--ids", default="", help="Необязательный список ID через запятую внутри выбранного split")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--warmup", default="dev01", help="ID примера development; пустая строка отключает прогрев")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/experiment")
    args = parser.parse_args(argv)
    if args.operation == "pipeline" and args.engine == "faster-whisper":
        args.engine = "our-indexed"
    if args.operation == "pipeline" and args.engine not in {"our", "our-indexed"}:
        parser.error("pipeline выполняется только через публичный Session нашего приложения")
    if args.engine == "our-indexed":
        args.indexed = True
    if args.repeat < 1:
        parser.error("--repeat должен быть >=1")
    dataset = manifest(args.manifest)
    items = select_items(dataset, args.split)
    if args.ids:
        chosen = set(args.ids.split(","))
        items = [item for item in items if item["id"] in chosen]
        if {item["id"] for item in items} != chosen:
            parser.error("Некоторые --ids отсутствуют в выбранном split")
    if args.operation == "inventory":
        inventory = []
        for item in select_items(dataset, "all") + select_items(dataset, "control"):
            path = ROOT / item["audio"]
            inventory.append({"id": item["id"], "split": item["split"], "exists": path.is_file(),
                              "seconds": audio_duration(path) if path.is_file() else None,
                              "sha256": sha256(path) if path.is_file() else None,
                              "expected_facts": len(item["expected"])})
        print(json.dumps({"manifest_sha256": sha256(args.manifest), "data": inventory}, ensure_ascii=False, indent=2))
        return 0
    if (args.out / "results.json").exists():
        parser.error("Каталог уже содержит results.json; выберите новый --out, чтобы сохранить прежний опыт")
    args.out.mkdir(parents=True, exist_ok=True)
    prompt = args.prompt_file.read_text(encoding="utf-8-sig") if args.prompt_file else PROMPT
    if args.operation in {"extract", "pipeline"} and args.engine in {"our", "our-indexed"}:
        if args.prompt_file:
            parser.error("Наш адаптер использует собственную производственную подсказку; --prompt-file неприменим")
        if args.engine == "our-indexed":
            from audio_mvp.providers_indexed import PROMPT as application_prompt
        else:
            from audio_mvp.providers import PROMPT as application_prompt
        prompt = application_prompt
    elif args.operation == "extract" and args.indexed and not args.prompt_file:
        prompt = indexed_analog_prompt()
    (args.out / "prompt.txt").write_text(prompt, encoding="utf-8")
    metadata = {"started_at": datetime.now(timezone.utc).isoformat(), "operation": args.operation,
                "arguments": vars(args), "environment": environment(), "manifest_sha256": sha256(args.manifest),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "dataset_validation": dataset["validation_status"],
                "scope": "actual public Session.process_file, audio validation+ASR+extraction; time until Document ready, excludes GUI painting and prestarted LLM startup" if args.operation == "pipeline" else "ASR component comparison" if args.operation == "asr" else
                ("actual Buzz AI Summary component; application UI not executed" if args.engine == "buzz" else
                 "actual OMS API note generation; application UI not executed" if args.engine == "oms" else "our local extraction adapter"),
                "scoring": "v2: exact normalized quote equality, correct section, one-to-one, source grounding; not independent semantic or clinical assessment",
                "script_sha256": sha256(Path(__file__)),
                "application_provider_sha256": sha256(ROOT / "audio_mvp/providers.py")}
    runtime_manifest = ROOT / ".runtime/manifest.json"
    if runtime_manifest.is_file():
        metadata["runtime_manifest_sha256"] = sha256(runtime_manifest)
    metadata["runtime_manifests"] = {p.name: sha256(p) for p in (ROOT / ".runtime").glob("manifest*.json")}
    if args.indexed:
        metadata["indexed_provider_sha256"] = sha256(ROOT / "audio_mvp/providers_indexed.py")
        metadata["indexed_note"] = "Модель выбирает номера одного общего списка предложений. Внешний декодер сопоставляет номера исходным цитатам. Это одинаковая техническая подготовка данных для всех конфигураций, не новая функция внутри аналога."
    if args.model_server:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(args.model_server.rstrip("/") + "/props", timeout=10) as response:
                metadata["model_server_props"] = json.loads(response.read().decode("utf-8"))
        except Exception as error:
            metadata["model_server_inspection_error"] = repr(error)
    if args.operation == "asr":
        metadata["recognition_settings"] = ({"device": "cpu", "compute_type": "int8", "language": "ru",
                                             "beam_size": 5, "vad_filter": True,
                                             "condition_on_previous_text": False, "cpu_threads": args.threads}
                                            if args.engine == "faster-whisper" else
                                            {"sample_rate": 16000, "grammar": None, "chunk_frames": 4000})
    else:
        metadata["generation_settings"] = ({"temperature": 0, "seed": 42, "max_tokens": 256 if args.indexed else 1024,
                                             "response_format": "application JSON schema", "source_quote_filter": True}
                                            if args.engine in {"our", "our-indexed"} else
                                            {"temperature": 0.2, "seed": "not specified by adapter", "max_tokens": "not specified by adapter",
                                             "response_format": "json_object", "timeout_seconds": 60, "source_quote_filter": False}
                                            if args.engine == "oms" else
                                            {"temperature": "not specified by adapter", "seed": "not specified by adapter",
                                             "max_tokens": "not specified by adapter", "response_format": "not specified by adapter",
                                             "timeout_seconds": 120, "source_quote_filter": False})
        metadata["comparison_limit"] = "Штатные адаптеры используют разные схемы, подсказки и настройки. Это сравнение конфигураций компонентов на одной задаче, не изолированное сравнение архитектур и не оценка всего GUI."
    save_json(args.out / "environment.json", metadata)
    load_start = time.perf_counter()
    try:
        adapter = (asr_adapter(args.engine, args.model, args.threads) if args.operation == "asr" else
                   pipeline_adapter(args) if args.operation == "pipeline" else extraction_adapter(args, prompt))
    except Exception as error:
        metadata["setup_error"] = repr(error)
        save_json(args.out / "environment.json", metadata)
        save_json(args.out / "results.json", {"runs": [], "summary": {"attempted": 0, "setup_error": repr(error)}})
        print(f"Ошибка подготовки эксперимента: {error}", file=sys.stderr)
        return 2
    metadata["adapter_load_seconds"] = time.perf_counter() - load_start
    if args.warmup:
        warmup = next((i for i in dataset["items"] if i["id"] == args.warmup and i["split"] == "development"), None)
        if warmup is None:
            parser.error("Прогрев допускается только на development ID")
        started = time.perf_counter()
        try:
            # Прогрев LLM использует development-эталон, даже если итоговый вход
            # берётся из ASR-транскрипций: оценочные примеры не используются.
            warmup_input = ROOT / warmup["audio"] if args.operation in {"asr", "pipeline"} else warmup["text"]
            warmup_output = adapter(warmup_input)
            metadata["warmup"] = {"id": warmup["id"], "input": "development audio" if args.operation in {"asr", "pipeline"} else "development reference text",
                                  "seconds": time.perf_counter() - started, "status": "completed"}
            save_json(args.out / "warmup.json", warmup_output)
        except Exception as error:
            metadata["warmup"] = {"id": warmup["id"], "seconds": time.perf_counter() - started, "status": "failed", "error": repr(error)}
    save_json(args.out / "environment.json", metadata)
    runs = []
    for item in items:
        for repetition in range(1, args.repeat + 1):
            started = time.perf_counter()
            run = {"id": item["id"], "split": item["split"], "repetition": repetition, "status": "failed"}
            try:
                if args.operation == "asr":
                    audio = ROOT / item["audio"]
                    text, raw = adapter(audio)
                    run["seconds"] = time.perf_counter() - started
                    save_json(args.out / "raw" / f"{item['id']}-{repetition}.json", raw)
                    run["duration_seconds"] = audio_duration(audio)
                    run["rtf"] = run["seconds"] / run["duration_seconds"]
                    run["audio_sha256"] = sha256(audio)
                    run["transcript"] = text
                    run["metrics"] = wer_counts(item["text"], text)
                    output = args.out / "transcripts" / f"{item['id']}.txt"
                    output.parent.mkdir(exist_ok=True)
                    output.write_text(text, encoding="utf-8")
                elif args.operation == "pipeline":
                    audio = ROOT / item["audio"]
                    raw = adapter(audio)
                    run["seconds"] = time.perf_counter() - started
                    save_json(args.out / "raw" / f"{item['id']}-{repetition}.json", raw)
                    text = raw["document"]["transcript"]
                    run["transcript"] = text
                    run["duration_seconds"] = audio_duration(audio)
                    run["rtf"] = run["seconds"] / run["duration_seconds"]
                    run["audio_sha256"] = sha256(audio)
                    run["wer_metrics"] = wer_counts(item["text"], text)
                    run["predictions"] = extracted_items(raw)
                    run["metrics"] = score_extraction(item["expected"], run["predictions"], text)
                    output = args.out / "transcripts" / f"{item['id']}.txt"
                    output.parent.mkdir(exist_ok=True)
                    output.write_text(text, encoding="utf-8")
                else:
                    text = input_text(item, args)
                    raw = adapter(text)
                    run["seconds"] = time.perf_counter() - started
                    save_json(args.out / "raw" / f"{item['id']}-{repetition}.json", raw)
                    run["input_text"] = text
                    # Качество всей цепочки всё равно сверяется с эталоном исходного диалога.
                    predictions = extracted_items(raw)
                    run["predictions"] = predictions
                    run["metrics"] = score_extraction(item["expected"], predictions, text)
                save_json(args.out / "raw" / f"{item['id']}-{repetition}.json", raw)
                run["status"] = "completed"
            except Exception as error:
                run["seconds"] = time.perf_counter() - started
                run["error"] = repr(error)
                if getattr(error, "model_response", None) is not None:
                    save_json(args.out / "raw" / f"{item['id']}-{repetition}-rejected.json",
                              {"error": repr(error), "model_response": error.model_response})
            runs.append(run)
            save_json(args.out / "results.json", {"runs": runs, "summary": summarize(runs, args.operation)})
            print(f"{args.engine} {item['id']} #{repetition}: {run['status']} {run['seconds']:.3f}s", flush=True)
    print(json.dumps(summarize(runs, args.operation), ensure_ascii=False, indent=2))
    return 0 if all(run["status"] == "completed" for run in runs) else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
