"""Одинаковый полный аудиовход для GUI, CLI и эксперимента."""
from pathlib import Path
from typing import Final
import av
import numpy as np
from .model import AppError

SAMPLE_RATE: Final = 16000


def read_audio(path):
    """Декодировать весь первый аудиопоток в mono PCM16, затем float32 [-1, 1]."""
    try:
        chunks: list[np.ndarray] = []
        resampler = av.AudioResampler(format='s16', layout='mono', rate=SAMPLE_RATE)
        with av.open(str(Path(path))) as source:
            if not source.streams.audio:
                raise ValueError('Нет аудиопотока')
            for frame in source.decode(audio=0):
                chunks.extend(out.to_ndarray().reshape(-1).copy() for out in resampler.resample(frame))
            chunks.extend(out.to_ndarray().reshape(-1).copy() for out in resampler.resample(None))
        if not chunks:
            raise ValueError('Пустое аудио')
        samples = np.concatenate(chunks).astype(np.float32) / 32768.0
        if not np.isfinite(samples).all():
            raise ValueError('Недопустимые значения сигнала')
        return samples
    except Exception as exc:
        raise AppError('Не удалось полностью прочитать аудиофайл. Проверьте формат и содержимое.') from exc


def audio_chunks(samples):
    """Полное непересекающееся покрытие; максимум 20 секунд на фрагмент.

    Граница длинного фрагмента — в области минимальной энергии между 12 и 20 с.
    Это не VAD: ни один отсчёт не удаляется.
    """
    maximum, minimum, window, step = (20 * SAMPLE_RATE, 12 * SAMPLE_RATE,
                                       SAMPLE_RATE // 5, SAMPLE_RATE // 10)
    start = 0
    while start < len(samples):
        end = min(start + maximum, len(samples))
        if end < len(samples):
            end = min(end, len(samples) - 6 * SAMPLE_RATE)
            positions = range(start + minimum, end - window + 1, step)
            quietest = min(positions, key=lambda p: float(np.mean(samples[p:p + window] ** 2)))
            end = quietest + window // 2
        yield start, end, samples[start:end]
        start = end
