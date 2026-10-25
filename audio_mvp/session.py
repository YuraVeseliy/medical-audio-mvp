"""Публичные операции одного диалога; внешние модели передаются при создании."""
from pathlib import Path
import threading
from .model import AppError, Document, SECTIONS


class Session:
    def __init__(self, recognizer, extractor):
        self.recognizer = recognizer
        self.extractor = extractor
        self.state = "ready"
        self.document = None
        self.dirty = False
        self._lock = threading.Lock()

    def process_file(self, path):
        with self._lock:
            if self.state in {"processing", "recording"}:
                raise AppError("Дождитесь завершения текущей операции.")
            if self.dirty:
                raise AppError("Сохраните правки или явно начните новый сеанс.")
            self.state = "processing"
        try:
            return self._process_file(path)
        except Exception:
            self.state = "error"
            raise

    def _process_file(self, path):
        path = Path(path)
        try:
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError("Файл отсутствует или пуст")
            if path.suffix.lower() not in {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4"}:
                raise ValueError("Формат не поддерживается")
            import av
            with av.open(str(path)) as source:
                if not source.streams.audio or next(source.decode(audio=0), None) is None:
                    raise ValueError("Нет декодируемых аудиоданных")
        except ImportError as exc:
            raise AppError("Не установлен аудиодекодер. Выполните подготовку проекта.") from exc
        except Exception as exc:
            raise AppError("Не удалось открыть аудиофайл: проверьте формат и содержимое.") from exc
        self.state = "processing"
        try:
            segments = self.recognizer.transcribe(Path(path))
            text = " ".join(s.text.strip() for s in segments if s.text.strip())
            if not text:
                raise AppError("Речь не распознана. Проверьте запись или выберите другой файл.")
            # Успешная транскрипция остаётся доступна даже при отказе LLM.
            # Новый объект не позволяет принять прежний результат за новый.
            self.document = Document(text, [], {key: "" for key in SECTIONS},
                                     ["Извлечение сведений ещё не завершено."], str(path))
            self.dirty = True
            evidence, warnings = self.extractor.extract(text)
        except Exception as exc:
            self.state = "error"
            if self.document is not None and self.dirty:
                self.document.warnings = ["Транскрипция сохранена, но автоматическое распределение не выполнено. Заполните разделы вручную или сохраните текст перед повторной обработкой."]
            if isinstance(exc, AppError):
                raise
            raise AppError(f"Не удалось обработать диалог ({type(exc).__name__}). Повторите попытку.") from exc
        sections = {key: "\n".join(e.quote for e in evidence if e.section == key)
                    for key in SECTIONS}
        self.document = Document(text, evidence, sections, warnings, str(path))
        self.state = "result"
        self.dirty = False
        return self.document

    def edit(self, section, text):
        if self.document is None or section not in SECTIONS:
            raise AppError("Сначала получите результат и выберите раздел.")
        if self.state in {"processing", "recording"}:
            raise AppError("Дождитесь завершения текущей операции.")
        if self.document.sections[section] != text:
            self.document.sections[section] = text
            self.dirty = True

    def save(self, path):
        from .storage import save_document
        if self.state in {"processing", "recording"}:
            raise AppError("Дождитесь завершения текущей операции.")
        if self.document is None:
            raise AppError("Нет результата для сохранения.")
        save_document(self.document, path)
        self.dirty = False

    def open(self, path):
        from .storage import load_document
        if self.state in {"processing", "recording"}:
            raise AppError("Дождитесь завершения текущей операции.")
        if self.dirty:
            raise AppError("Сохраните правки или явно начните новый сеанс.")
        document = load_document(path)
        self.document = document
        self.state = "result"
        self.dirty = False
        return document

    def new(self, force=False):
        if self.state in {"processing", "recording"}:
            raise AppError("Дождитесь завершения текущей операции.")
        if self.dirty and not force:
            raise AppError("В сеансе есть несохранённые правки. Сохраните их или подтвердите новый сеанс.")
        self.document = None
        self.state = "ready"
        self.dirty = False
