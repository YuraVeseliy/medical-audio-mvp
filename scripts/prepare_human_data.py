"""Отбор, проверка, соединение и общий формат личного учебного набора."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import wave
import av

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audio_mvp.audio import read_audio
from audio_mvp.provenance import source_version


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio-dir', type=Path, required=True)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--confirmed-references', action='store_true',
                        help='Указать только после личного прослушивания и сверки TXT со звуком')
    parser.add_argument('--executor', default='unspecified')
    parser.add_argument('--reference-verifier', default=None)
    parser.add_argument('--device', default='диктофон; точная модель и микрофон не уточнены')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Папка результата уже существует. Новая версия данных требует новой папки.')
    expected = ['dev01', 'dev02', *[f'dialog{i:02}' for i in range(1, 7)], 'control01', 'control02']
    sources = {}
    for path in args.audio_dir.iterdir():
        if path.is_file() and path.suffix.lower() in {'.wav','.m4a','.mp3','.flac','.ogg','.mp4','.mkv'}:
            key = path.stem.lower()
            if key in sources:
                parser.error('Несколько аудиозаписей с ID ' + key)
            sources[key] = path
    if set(sources) != set(expected):
        parser.error('Нужны ровно ID dev01/dev02, dialog01..06, control01/control02')
    references = {path.stem.lower(): path for path in args.reference_dir.glob('*.txt')}
    if any(key not in references for key in expected):
        parser.error('Для каждого аудио нужен один TXT с таким же ID')
    args.out.mkdir(parents=True)
    for folder in ['raw','prepared','references']:
        (args.out / folder).mkdir()
    items = []
    for key in expected:
        path, ref_path = sources[key], references[key]
        data, ref = path.read_bytes(), ref_path.read_bytes()
        assert ref.decode('utf-8-sig').strip(), 'Пустой эталон: ' + key
        (args.out / 'raw' / path.name).write_bytes(data)
        # Единая UTF-8 без BOM; слова и пунктуация эталона не исправляются.
        ref = ref.decode('utf-8-sig').encode('utf-8')
        (args.out / 'references' / (key + '.txt')).write_bytes(ref)
        samples = read_audio(path)
        with av.open(str(path)) as container:
            stream = container.streams.audio[0]
            input_format = {'sample_rate': stream.codec_context.sample_rate,
                            'channels': stream.codec_context.channels,
                            'codec': stream.codec_context.name}
        pcm = (samples * 32768).astype('<i2').tobytes()
        prepared = args.out / 'prepared' / (key + '.wav')
        with wave.open(str(prepared), 'wb') as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(16000)
            out.writeframes(pcm)
        split = 'development' if key.startswith('dev') else 'selection' if key.startswith('dialog') else 'final_control'
        items.append({'id': key, 'split': split, 'source_name': path.name, 'source_sha256': sha(data),
                      'raw': 'raw/' + path.name, 'reference': 'references/' + key + '.txt',
                      'reference_sha256': sha(ref), 'prepared': 'prepared/' + key + '.wav',
                      'prepared_sha256': sha(prepared.read_bytes()), 'decoded_pcm_sha256': sha(pcm),
                      'duration_s': len(samples)/16000, 'input': input_format,
                      'prepared_format': {'sample_rate': 16000, 'channels': 1, 'sample_width_bytes': 2},
                      'reference_status': 'confirmed_by_recording_author' if args.confirmed_references else 'pending'})
    duplicate_groups = []
    for i, item in enumerate(items):
        for other in items[i+1:]:
            if item['decoded_pcm_sha256'] == other['decoded_pcm_sha256']:
                duplicate_groups.append([item['id'],other['id']])
    manifest = {'version': 'human-audio-v1', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                **source_version(ROOT), 'device_reported_by_user': args.device, 'executor': args.executor,
                'reference_verification': {'status': 'confirmed_by_recording_author' if args.confirmed_references else 'pending',
                                           'verified_by': (args.reference_verifier or args.executor) if args.confirmed_references else None,
                                           'independent_second_reviewer': False},
                'processing': 'whole audio -> mono PCM16 16kHz; no denoising, trimming or gain adjustment',
                'items': items, 'duplicate_decoded_audio': duplicate_groups,
                'preparation_successful': not duplicate_groups,
                'dataset_contents_sha256': sha(json.dumps(items, sort_keys=True, ensure_ascii=False).encode('utf-8'))}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    if duplicate_groups:
        raise SystemExit('Обнаружены одинаковые записи. Сначала решите вопрос дубликатов: ' + str(duplicate_groups))
    print(json.dumps({'prepared': len(items), 'output': str(args.out),
                      'references_confirmed': args.confirmed_references}, ensure_ascii=True))


if __name__ == '__main__':
    main()
