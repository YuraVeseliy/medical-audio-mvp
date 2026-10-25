"""Сохранение пользовательского текста; атомарная замена файла после записи."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import uuid

from .model import AppError, Document, Evidence, SECTIONS


def _atomic_text(path, text):
    target = Path(path)
    temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except OSError as exc:
        raise AppError("Не удалось сохранить файл. Текущий текст остался в приложении.") from exc
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def save_document(document, path):
    payload = {"format": "medical-audio-mvp", "version": 1, "document": asdict(document)}
    _atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def load_document(path):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != "medical-audio-mvp" or payload.get("version") != 1:
            raise ValueError("Неизвестная версия сеанса")
        item = payload["document"]
        if not isinstance(item["transcript"], str) or not isinstance(item["source"], str):
            raise ValueError("Неверный текст")
        if set(item["sections"]) != set(SECTIONS) or not all(
                isinstance(value, str) for value in item["sections"].values()):
            raise ValueError("Неверные разделы")
        evidence = []
        for row in item["evidence"]:
            e = Evidence(**row)
            if (e.section not in SECTIONS or not isinstance(e.quote, str)
                    or type(e.start) is not int or type(e.end) is not int
                    or e.start < 0 or e.end <= e.start
                    or item["transcript"][e.start:e.end] != e.quote):
                raise ValueError("Недостоверная ссылка на источник")
            evidence.append(e)
        warnings = item["warnings"]
        if not isinstance(warnings, list) or not all(isinstance(x, str) for x in warnings):
            raise ValueError("Неверные предупреждения")
        return Document(item["transcript"], evidence, dict(item["sections"]), warnings, item["source"])
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, AttributeError) as exc:
        raise AppError("Не удалось открыть сеанс: файл повреждён или имеет неизвестный формат.") from exc


def export_text(document, path):
    parts = ["Учебный черновик. Проверьте сведения по исходной записи.", ""]
    for key, title in SECTIONS.items():
        parts.extend([title, document.sections.get(key, "") or "Сведения не выделены.", ""])
    parts.extend(["Исходная транскрипция", document.transcript])
    _atomic_text(path, "\n".join(parts) + "\n")
