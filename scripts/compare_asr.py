"""Общий воспроизводимый эксперимент через тот же Session, что использует GUI."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audio_mvp.metrics import word_errors, words
from audio_mvp.provenance import source_version, hardware
from audio_mvp.providers import ASRRecognizer, MODELS, MODEL_BY_ID, verify_installation
from audio_mvp.session import Session
from audio_mvp.storage import save_json as write
from audio_mvp.model import AppError


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_dataset(folder):
    manifest_path = folder / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('reference_verification', {}).get('status') != 'confirmed_by_recording_author':
        raise ValueError('Перед WER подтвердите соответствие TXT звуку; сценарий чтения сам по себе не эталон.')
    if manifest.get('duplicate_decoded_audio') or manifest.get('audit', {}).get('duplicate_decoded_audio'):
        raise ValueError('Набор содержит дубликаты звука. Сначала устраните пересечение групп.')
    if manifest.get('preparation_successful') is False:
        raise ValueError('Подготовка набора не завершилась успешно.')
    ids, audio_hashes = set(), set()
    for item in manifest['items']:
        if item['id'] in ids:
            raise ValueError('Дубликат ID в наборе: ' + item['id'])
        ids.add(item['id'])
        for path_key, hash_key in [('prepared', 'prepared_sha256'), ('reference', 'reference_sha256')]:
            path = (folder / item[path_key]).resolve()
            if not path.is_relative_to(folder.resolve()) or sha(path) != item[hash_key]:
                raise ValueError('Набор изменён или содержит неверный путь: ' + item['id'])
        from audio_mvp.audio import read_audio
        pcm = (read_audio(folder / item['prepared']) * 32768).astype('<i2').tobytes()
        decoded_hash = hashlib.sha256(pcm).hexdigest()
        if decoded_hash in audio_hashes:
            raise ValueError('В разных случаях обнаружено одинаковое аудио: ' + item['id'])
        audio_hashes.add(decoded_hash)
    return manifest


def validate_control(previous, source_hash, dataset_hash, threads, installation):
    model_id = previous.get('chosen_model')
    if previous.get('split') != 'selection' or not model_id:
        raise ValueError('Нет зафиксированного выбора модели на selection')
    if previous.get('source_sha256') != source_hash or previous.get('dataset_manifest_sha256') != dataset_hash:
        raise ValueError('Код или набор изменился после выбора модели')
    if previous.get('threads_requested') != threads:
        raise ValueError('Изменились настройки числа потоков')
    earlier = next((row for row in previous.get('models', []) if row['model_id'] == model_id), None)
    if (not earlier or installation.get('model_id') != model_id
            or earlier.get('installation_fingerprint') != installation.get('fingerprint')):
        raise ValueError('Изменились фактические веса или конфигурация модели')
    return model_id


def recover_worker_result(folder, model_id):
    """Восстановить завершённые строки и пометить прерванный прогон неполным."""
    def read_optional(name, default):
        path = folder / name
        try:
            return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else default
        except (OSError, ValueError):
            return default
    checkpoint = read_optional('checkpoint.json', {})
    return {**checkpoint, 'model_id': model_id, 'status': 'failed',
            'rows': read_optional('rows.json', []),
            'memory_measurement_complete': False,
            'error': 'Worker interrupted; completed rows preserved. See console.log; later memory peak unknown.'}


def run_worker(args):
    import psutil
    folder = args.out / args.worker
    folder.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    peak = [process.memory_info().rss]
    stopped = threading.Event()
    def sample():
        while not stopped.wait(0.02):
            peak[0] = max(peak[0], process.memory_info().rss)
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    rows = []
    recognizer = ASRRecognizer(args.worker, args.runtime, args.threads)
    manifest = validate_dataset(args.dataset)
    try:
        load_seconds = recognizer.load()
        development = next(item for item in manifest['items'] if item['id'] == 'dev01')
        warmup = Session(recognizer).process_file(args.dataset / development['prepared'])
        write(folder / 'warmup.json', asdict(warmup))
        write(folder / 'checkpoint.json', {'model_load_seconds': load_seconds,
              'peak_worker_rss_bytes': peak[0], 'last_completed_count': 0})
        selected = [item for item in manifest['items'] if item['split'] == args.split]
        for repeat in range(1, args.repeats + 1):
            for item in selected:
                reference = (args.dataset / item['reference']).read_text(encoding='utf-8')
                row = {'id': item['id'], 'repeat': repeat, 'model_id': args.worker,
                       'audio_sha256': item['prepared_sha256'], 'reference_sha256': item['reference_sha256']}
                started = time.perf_counter()
                try:
                    document = Session(recognizer).process_file(args.dataset / item['prepared'])
                    elapsed = time.perf_counter() - started
                    hypothesis = document.transcript
                    write(folder / f"{item['id']}-{repeat:02}.json", asdict(document))
                    row.update({'successful': True, 'seconds': elapsed, 'hypothesis': hypothesis,
                                'duration_seconds': document.metrics['duration_seconds'],
                                'chunks': document.metrics['chunks']})
                except Exception as error:
                    hypothesis = ''
                    row.update({'successful': False, 'seconds': time.perf_counter() - started,
                                'hypothesis': '', 'duration_seconds': item['duration_s'],
                                'error_type': type(error).__name__, 'error': str(error)})
                row.update(word_errors(reference, hypothesis))
                row['rtf'] = row['seconds'] / row['duration_seconds']
                rows.append(row)
                write(folder / 'rows.json', rows)
                write(folder / 'checkpoint.json', {'model_load_seconds': load_seconds,
                      'peak_worker_rss_bytes': peak[0], 'last_completed_count': len(rows)})
                print(json.dumps({'model': args.worker, 'id': item['id'], 'repeat': repeat,
                                  'ok': row['successful'], 'wer': row['wer'], 'seconds': row['seconds']}), flush=True)
        result = {'model_id': args.worker, 'status': 'completed', 'rows': rows,
                  'model_load_seconds': load_seconds, 'peak_worker_rss_bytes': peak[0],
                  'memory_scope': 'peak RSS of recognition worker including Python, imports, model load and all files; sampled every 20ms'}
    except Exception as error:
        (folder / 'failure.log').write_text(traceback.format_exc(), encoding='utf-8')
        result = {'model_id': args.worker, 'status': 'failed', 'rows': rows,
                  'error_type': type(error).__name__, 'error': str(error), 'peak_worker_rss_bytes': peak[0]}
    finally:
        stopped.set()
        sampler.join(2)
        recognizer.close()
    write(folder / 'worker.json', result)
    return 0 if result['status'] == 'completed' else 1


def aggregate(result, expected_count, expected_words):
    rows = result.get('rows', [])
    errors = sum(row['errors'] for row in rows)
    completed = len(rows)
    # Отсутствующие строки не исчезают из знаменателя: это полные пропуски эталона.
    seen_words = sum(row['reference_words'] for row in rows)
    errors += expected_words - seen_words
    return {'model_id': result['model_id'], 'status': result['status'],
            'attempts_recorded': completed, 'expected_attempts': expected_count,
            'successful': sum(row['successful'] for row in rows),
            'eligible': result['status'] == 'completed' and completed == expected_count and all(row['successful'] for row in rows),
            'errors': errors, 'reference_words': expected_words, 'wer': errors / expected_words,
            'median_seconds': statistics.median(row['seconds'] for row in rows) if rows else None,
            'median_rtf': statistics.median(row['rtf'] for row in rows) if rows else None,
            'peak_worker_rss_bytes': result.get('peak_worker_rss_bytes'),
            'model_load_seconds': result.get('model_load_seconds'), 'error': result.get('error')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, default=ROOT / '.runtime')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--models', nargs='+', default=['all'])
    parser.add_argument('--split', choices=['development', 'selection', 'final_control'], default='selection')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--executor', default='unspecified')
    parser.add_argument('--control-from', type=Path)
    parser.add_argument('--worker', choices=list(MODEL_BY_ID), help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.dataset, args.out, args.runtime = args.dataset.resolve(), args.out.resolve(), args.runtime.resolve()
    if args.repeats < 1 or args.threads < 1:
        parser.error('Число повторов и потоков должно быть положительным')
    if args.worker:
        return run_worker(args)
    if args.out.exists():
        parser.error('Папка результата уже существует. Укажите новое имя, чтобы сохранить предыдущий прогон.')
    manifest = validate_dataset(args.dataset)
    version = source_version(ROOT)
    model_ids = [item.id for item in MODELS] if args.models == ['all'] else args.models
    if args.split == 'final_control':
        if not args.control_from:
            parser.error('Для контроля нужен --control-from с итогом выбора на selection')
        previous = json.loads(args.control_from.read_text(encoding='utf-8'))
        try:
            installation = verify_installation(previous['chosen_model'], args.runtime)
            chosen = validate_control(previous, version['source_sha256'], sha(args.dataset / 'manifest.json'),
                                      args.threads, installation)
        except (ValueError, KeyError) as error:
            parser.error(str(error))
        model_ids = [chosen]
    if any(model not in MODEL_BY_ID for model in model_ids):
        parser.error('Неизвестная модель')
    entries = [item for item in manifest['items'] if item['split'] == args.split]
    if not entries:
        parser.error('В выбранной группе нет файлов')
    expected_words = sum(len(words(
        (args.dataset / item['reference']).read_text(encoding='utf-8'))) for item in entries) * args.repeats
    definitions = []
    for model_id in model_ids:
        installed = verify_installation(model_id, args.runtime)
        definitions.append({'model_id': model_id, 'installation_manifest_sha256': installed['manifest_sha256'],
                            'installation_fingerprint': installed['fingerprint'], 'installation': installed['manifest']})
    args.out.mkdir(parents=True)
    freeze = {'time_utc': datetime.now(timezone.utc).isoformat(), **version,
              'dataset_manifest_sha256': sha(args.dataset / 'manifest.json'),
              'dataset_contents_sha256': manifest['dataset_contents_sha256'],
              'split': args.split, 'ids': [item['id'] for item in entries],
              'independent_recordings': len(entries), 'repeats': args.repeats,
              'models': definitions, 'threads_requested': args.threads, 'executor': args.executor,
              'platform': platform.platform(), 'python': sys.version, 'hardware': hardware(),
              'packages': {name: importlib.metadata.version(name) for name in
                           ['av','numpy','faster-whisper','ctranslate2','vosk','onnx-asr','onnxruntime','psutil']},
              'selection_rule': 'eligible only: lowest aggregate WER, then median RTF, then peak worker RSS',
              'normalization': 'lowercase, yo/e, punctuation and whitespace; no semantic number or medical-word correction',
              'recording_device_reported': manifest['device_reported_by_user']}
    write(args.out / 'freeze.json', freeze)
    results = []
    for model_id in model_ids:
        folder = args.out / model_id
        folder.mkdir()
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', model_id,
                   '--dataset', str(args.dataset), '--runtime', str(args.runtime), '--out', str(args.out),
                   '--split', args.split, '--repeats', str(args.repeats), '--threads', str(args.threads)]
        print('Начат прогон: ' + model_id, flush=True)
        with (folder / 'console.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            try:
                process.wait(timeout=1200)
            except subprocess.TimeoutExpired:
                import psutil
                for child in psutil.Process(process.pid).children(recursive=True):
                    child.kill()
                process.kill()
                process.wait()
        record = folder / 'worker.json'
        result = json.loads(record.read_text(encoding='utf-8')) if record.is_file() else recover_worker_result(folder, model_id)
        if (sha(args.dataset / 'manifest.json') != freeze['dataset_manifest_sha256']
                or source_version(ROOT)['source_sha256'] != freeze['source_sha256']):
            write(args.out / 'invalidated.json', {'reason': 'Code or dataset changed during the run; no valid winner.'})
            raise ValueError('Код или набор изменился во время измерений. Прогон не пригоден для выбора.')
        results.append(aggregate(result, len(entries) * args.repeats, expected_words))
        print(json.dumps(results[-1]), flush=True)
    eligible = [row for row in results if row['eligible']]
    best = min(eligible, key=lambda r: (r['wer'], r['median_rtf'], r['peak_worker_rss_bytes'])) if eligible else None
    summary = {**freeze, 'results': results,
               'chosen_model': best['model_id'] if best and args.split == 'selection' else None,
               'control_model': model_ids[0] if args.split == 'final_control' else None,
               'clinical_acceptance': 'not evaluated; small single-reader fictional dataset',
               'status': 'completed' if all(row['eligible'] for row in results) else 'completed_with_failures'}
    write(args.out / 'summary.json', summary)
    lines = ['# Сравнение локальных ASR', '', f"Группа: {args.split}. Независимых записей: {len(entries)}. Повторов: {args.repeats}.", '',
             '| Модель | Завершено | WER, % | Медиана, с | Медиана RTF | Пик RSS, MiB |', '|---|---:|---:|---:|---:|---:|']
    for row in results:
        seconds = f"{row['median_seconds']:.3f}" if row['median_seconds'] is not None else '—'
        rtf = f"{row['median_rtf']:.3f}" if row['median_rtf'] is not None else '—'
        memory = f"{row['peak_worker_rss_bytes']/1048576:.1f}" if row['peak_worker_rss_bytes'] else '—'
        lines.append(f"| {row['model_id']} | {row['successful']}/{row['expected_attempts']} | {row['wer']*100:.2f} | {seconds} | {rtf} | {memory} |")
    lines += ['', 'Выбранная конфигурация: ' + str(summary['chosen_model']), '',
              'Прогоны одного файла не являются новыми независимыми записями. Сбои учитываются, а не исключаются из WER. Числа и варианты записи слов требуют отдельного разбора. Это не клиническая приёмка.']
    (args.out / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (AppError, ValueError) as error:
        print('Эксперимент не завершён: ' + str(error), file=sys.stderr)
        raise SystemExit(2)
