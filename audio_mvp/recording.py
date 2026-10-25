"""Запись начинается только по явному вызову start; библиотеки импортируются лениво."""
from __future__ import annotations

from pathlib import Path
import threading
from typing import Any

from .model import AppError


class AudioRecorder:
    def __init__(self) -> None:
        self.status = "idle"
        self._stream: Any = None
        self._chunks: list[Any] = []
        self._sample_rate = 16000
        self._lock = threading.Lock()
        self._warnings: list[str] = []

    @staticmethod
    def devices() -> list[dict[str, Any]]:
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            default = sd.default.device[0]
            return [{"index": index, "name": item["name"],
                     "default": index == default,
                     "samplerate": item["default_samplerate"]}
                    for index, item in enumerate(devices) if item["max_input_channels"] > 0]
        except ImportError as exc:
            raise AppError("Не установлены библиотеки записи. Выполните установку зависимостей.") from exc
        except Exception as exc:
            raise AppError("Не удалось получить список микрофонов.") from exc

    @property
    def warnings(self) -> list[str]:
        with self._lock:
            return list(self._warnings)

    def start(self, device: int | None = None) -> None:
        if self.status != "idle":
            raise AppError("Запись уже идёт или ожидает сохранения.")
        try:
            import sounddevice as sd
            info = sd.query_devices(device, "input")
            self._sample_rate = 16000
            try:
                sd.check_input_settings(device=device, channels=1,
                                        samplerate=self._sample_rate, dtype="float32")
            except sd.PortAudioError:
                self._sample_rate = int(info["default_samplerate"])
                sd.check_input_settings(device=device, channels=1,
                                        samplerate=self._sample_rate, dtype="float32")
            with self._lock:
                self._chunks = []
                self._warnings = []

            def callback(indata: Any, frames: int, timing: Any, status: Any) -> None:
                with self._lock:
                    self._chunks.append(indata.copy())
                    if status and str(status) not in self._warnings:
                        self._warnings.append(str(status))

            self._stream = sd.InputStream(device=device, samplerate=self._sample_rate,
                                          channels=1, dtype="float32", callback=callback)
            self._stream.start()
            self.status = "recording"
        except ImportError as exc:
            raise AppError("Не установлены библиотеки записи. Выполните установку зависимостей.") from exc
        except Exception as exc:
            if self._stream is not None:
                try:
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
            self.status = "idle"
            raise AppError("Не удалось начать запись. Проверьте микрофон и разрешение доступа.") from exc

    def capture(self) -> None:
        """Остановить захват сразу, оставив звук в памяти до выбора пути."""
        if self.status not in {"recording", "captured"}:
            raise AppError("Запись ещё не начата.")
        if self._stream is not None:
            stream, self._stream = self._stream, None
            failure: Exception | None = None
            try:
                stream.stop()
            except Exception as exc:
                failure = exc
            try:
                stream.close()
            except Exception as exc:
                failure = failure or exc
            self.status = "captured"
            if failure is not None:
                raise AppError("Не удалось штатно остановить микрофон; можно повторить сохранение полученного звука.") from failure
        self.status = "captured"

    def stop(self, path: str | Path) -> Path:
        self.capture()
        try:
            import numpy as np
            import soundfile as sf
            with self._lock:
                chunks = list(self._chunks)
            if not chunks:
                self.status = "idle"
                raise AppError("Микрофон не передал аудиоданные. Попробуйте записать ещё раз.")
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            audio = np.concatenate(chunks, axis=0)
            sf.write(str(target), audio, self._sample_rate, format="WAV", subtype="PCM_16")
            with self._lock:
                self._chunks = []
            self.status = "idle"
            return target
        except ImportError as exc:
            raise AppError("Не установлена библиотека сохранения звука.") from exc
        except AppError:
            raise
        except Exception as exc:
            # Буфер сохранён: stop можно повторить с доступным путём.
            raise AppError("Не удалось сохранить запись. Выберите доступный путь и повторите сохранение.") from exc

    def discard(self) -> None:
        """Явно отменить запись/несохранённый буфер, например при закрытии приложения."""
        if self._stream is not None:
            self._stream.abort()
            self._stream.close()
            self._stream = None
        with self._lock:
            self._chunks = []
        self.status = "idle"
