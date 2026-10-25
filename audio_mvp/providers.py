"""Восемь локальных конфигураций ASR; загрузка весов отделена от инференса."""
from dataclasses import dataclass
import gc
import json
from pathlib import Path
import time

from .audio import SAMPLE_RATE, audio_chunks, read_audio
from .model import AppError, Segment


@dataclass(frozen=True)
class ModelConfig:
    id: str
    label: str
    engine: str
    folder: str
    variant: str = ''


MODELS = (
    ModelConfig('whisper-base', 'Whisper base · CPU int8', 'whisper', 'faster-whisper-base'),
    ModelConfig('whisper-small', 'Whisper small · CPU int8', 'whisper', 'faster-whisper-small'),
    ModelConfig('whisper-medium', 'Whisper medium · CPU int8', 'whisper', 'faster-whisper-medium'),
    ModelConfig('vosk-small', 'Vosk small-ru-0.22', 'vosk', 'vosk-model-small-ru-0.22'),
    ModelConfig('vosk-large', 'Vosk ru-0.42', 'vosk', 'vosk-model-ru-0.42'),
    ModelConfig('gigaam-ctc', 'GigaAM v3 CTC · ONNX int8', 'gigaam', 'gigaam-v3-onnx', 'gigaam-v3-ctc'),
    ModelConfig('gigaam-rnnt', 'GigaAM v3 RNN-T · ONNX int8', 'gigaam', 'gigaam-v3-onnx', 'gigaam-v3-rnnt'),
    ModelConfig('gigaam-e2e-rnnt', 'GigaAM v3 E2E RNN-T · ONNX int8', 'gigaam', 'gigaam-v3-onnx', 'gigaam-v3-e2e-rnnt'),
)
MODEL_BY_ID = {item.id: item for item in MODELS}


class WhisperBackend:
    def __init__(self, path, threads):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(str(path), device='cpu', compute_type='int8',
                                  cpu_threads=threads, num_workers=1, local_files_only=True)

    def recognize(self, samples):
        segments, _ = self.model.transcribe(samples, language='ru', task='transcribe',
            beam_size=5, temperature=0, vad_filter=False, condition_on_previous_text=False)
        return ' '.join(segment.text.strip() for segment in segments if segment.text.strip())


class VoskBackend:
    def __init__(self, path, threads):
        from vosk import Model, SetLogLevel
        SetLogLevel(-1)
        self.model = Model(str(path))

    def recognize(self, samples):
        import numpy as np
        from vosk import KaldiRecognizer
        recognizer = KaldiRecognizer(self.model, SAMPLE_RATE)
        pcm = np.clip(np.rint(samples * 32768), -32768, 32767).astype('<i2').tobytes()
        texts = []
        for start in range(0, len(pcm), 8000):
            if recognizer.AcceptWaveform(pcm[start:start + 8000]):
                texts.append(json.loads(recognizer.Result()).get('text', ''))
        texts.append(json.loads(recognizer.FinalResult()).get('text', ''))
        return ' '.join(text for text in texts if text)


class GigaAMBackend:
    def __init__(self, path, threads, variant):
        import onnxruntime as ort
        import onnx_asr
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.model = onnx_asr.load_model(variant, path, quantization='int8', sess_options=options,
            providers=['CPUExecutionProvider'], preprocessor_config={'max_concurrent_workers': 1})

    def recognize(self, samples):
        return self.model.recognize(samples, sample_rate=SAMPLE_RATE)


class ASRRecognizer:
    """Общий путь распознавания. В тестах заменяется только внешняя модель."""
    def __init__(self, model_id='whisper-small', runtime=None, threads=8, backend=None):
        if model_id not in MODEL_BY_ID:
            raise AppError('Неизвестная конфигурация распознавания.')
        self.config = MODEL_BY_ID[model_id]
        self.model_id = model_id
        self.runtime = Path(runtime or Path(__file__).resolve().parents[1] / '.runtime')
        self.threads = threads
        self.backend = backend
        self.last_metrics = {}

    def load(self):
        if self.backend is not None:
            return 0.0
        path = self.runtime / 'models' / self.config.folder
        if not path.is_dir():
            raise AppError(f'Модель не установлена. Выполните: python scripts/setup_asr_models.py --model {self.model_id}')
        started = time.perf_counter()
        try:
            if self.config.engine == 'whisper':
                self.backend = WhisperBackend(path, self.threads)
            elif self.config.engine == 'vosk':
                self.backend = VoskBackend(path, self.threads)
            else:
                self.backend = GigaAMBackend(path, self.threads, self.config.variant)
        except ImportError as exc:
            raise AppError('Не установлены библиотеки выбранного распознавателя. Выполните setup.cmd.') from exc
        except Exception as exc:
            raise AppError(f'Не удалось загрузить {self.config.label}: {type(exc).__name__}. Проверьте установку модели.') from exc
        return time.perf_counter() - started

    def transcribe(self, path):
        load_seconds = self.load()
        started = time.perf_counter()
        samples = read_audio(path)
        segments, boundaries = [], []
        for start, end, chunk in audio_chunks(samples):
            text = self.backend.recognize(chunk).strip()
            boundaries.append({'start_sample': start, 'end_sample': end})
            if text:
                segments.append(Segment(start / SAMPLE_RATE, end / SAMPLE_RATE, text))
        self.last_metrics = {'model_id': self.model_id, 'label': self.config.label,
            'engine': self.config.engine, 'cpu_threads_requested': self.threads,
            'duration_seconds': len(samples) / SAMPLE_RATE,
            'processing_seconds': time.perf_counter() - started,
            'model_load_seconds_this_call': load_seconds,
            'sample_rate': SAMPLE_RATE, 'sample_count': len(samples), 'chunks': boundaries,
            'timing_scope': 'full decoding, common chunking and ASR; model loading measured separately',
            'timestamp_scope': 'common audio chunks, not word-level alignment'}
        return segments

    def close(self):
        self.backend = None
        gc.collect()
