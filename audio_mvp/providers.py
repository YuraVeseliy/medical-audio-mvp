"""Восемь локальных конфигураций ASR; загрузка весов отделена от инференса."""
from dataclasses import dataclass
import gc
import hashlib
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


_CATALOG = json.loads((Path(__file__).resolve().parents[1] / 'configs/asr-models.json').read_text(encoding='utf-8'))
MODELS = tuple(ModelConfig(id=model_id, **{key: definition[key] for key in ('label','engine','folder','variant')})
               for model_id, definition in _CATALOG['models'].items())
MODEL_BY_ID = {item.id: item for item in MODELS}


def default_model_id() -> str:
    """Предпочтение интерфейса; в эксперименте модель всегда задана явно."""
    path = Path(__file__).resolve().parents[1] / 'configs/preferred-model.json'
    try:
        model_id = json.loads(path.read_text(encoding='utf-8')).get('model_id')
        if isinstance(model_id, str) and model_id in MODEL_BY_ID:
            return model_id
    except (OSError, ValueError, AttributeError):
        pass
    return 'whisper-small'


def verify_installation(model_id, runtime):
    """Подтвердить фактические локальные файлы и вернуть снимок происхождения."""
    runtime = Path(runtime).resolve()
    path = runtime / 'manifests' / (model_id + '.json')
    try:
        raw = path.read_bytes()
        record = json.loads(raw)
        if record['model_id'] != model_id or record['definition'] != _CATALOG['models'][model_id]:
            raise ValueError('Каталог и установленная конфигурация отличаются')
        files = record['installed_files']
        if not files:
            raise ValueError('Нет контрольных сумм фактических файлов')
        expected = {_CATALOG['artifacts'][key]['destination'] for key in record['definition']['artifacts']
                    if _CATALOG['artifacts'][key]['destination'].startswith('models/')}
        if expected and {item['path'] for item in files} != expected:
            raise ValueError('Список проверяемых файлов не соответствует каталогу')
        model_root = (runtime / 'models' / MODEL_BY_ID[model_id].folder).resolve()
        for item in files:
            target = (runtime / item['path']).resolve()
            if not target.is_relative_to(model_root) or target.stat().st_size != item['bytes']:
                raise ValueError('Изменён путь или размер установленного файла')
            with target.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != item['sha256']:
                    raise ValueError('Контрольная сумма установленного файла не совпадает')
        fingerprint = hashlib.sha256(json.dumps({'definition': record['definition'], 'files': files},
                                                sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()
        return {'model_id': model_id, 'manifest_sha256': hashlib.sha256(raw).hexdigest(),
                'fingerprint': fingerprint, 'manifest': record}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise AppError(f'Установка модели {model_id} не подтверждена ({exc}). Повторите setup_asr_models.py --model {model_id}.') from exc


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
        if not isinstance(threads, int) or threads < 1:
            raise AppError('Число потоков должно быть положительным целым числом.')
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
        verify_installation(self.model_id, self.runtime)
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
