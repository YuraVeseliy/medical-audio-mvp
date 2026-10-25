"""Отдельная установка локальных моделей по закреплённому каталогу."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time
from urllib import request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def open_url(url, headers=None):
    opener = request.build_opener(request.ProxyHandler({}))
    return opener.open(request.Request(url, headers=headers or {}), timeout=90)


def digest_file(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def valid_file(path, item):
    if not path.is_file() or (item.get('size') is not None and path.stat().st_size != item['size']):
        return False
    if item.get('sha256'):
        return digest_file(path) == item['sha256']
    if item.get('git_blob_sha1'):
        data = path.read_bytes()
        return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == item['git_blob_sha1']
    return False


def download(item, runtime):
    target = runtime / item['destination']
    target.parent.mkdir(parents=True, exist_ok=True)
    if valid_file(target, item):
        return target
    size = item.get('size')
    suffix = '&' if '?' in item['url'] else '?'
    if size is None:
        with open_url(item['url'], {'Range': 'bytes=0-0'}) as response:
            content_range = response.headers.get('Content-Range')
            size = int(content_range.split('/')[-1]) if content_range else int(response.headers['Content-Length'])
    folder = runtime / 'downloads/asr-parts' / hashlib.sha256(item['url'].encode()).hexdigest()[:16]
    folder.mkdir(parents=True, exist_ok=True)
    spans = [(start, min(start + 2 * 1024 * 1024, size) - 1) for start in range(0, size, 2 * 1024 * 1024)]

    def part(span):
        start, end = span
        path = folder / f'{start}-{end}.part'
        if path.is_file() and path.stat().st_size == end - start + 1:
            return path
        partial = path.with_suffix('.partial')
        for attempt in range(12):
            try:
                offset = partial.stat().st_size if partial.is_file() else 0
                if offset == end - start + 1:
                    partial.replace(path)
                    return path
                if offset > end - start + 1:
                    raise ValueError('Недопустимый размер частичной загрузки')
                first = start + offset
                url = item['url'] + suffix + f'download=true&part={first}-{end}&nonce={time.time_ns()}'
                with open_url(url, {'Range': f'bytes={first}-{end}'}) as response:
                    if not (response.status == 200 and first == 0 and end == size - 1):
                        if response.status != 206 or response.headers.get('Content-Range') != f'bytes {first}-{end}/{size}':
                            raise ValueError('Сервер не подтвердил диапазон загрузки')
                    with partial.open('ab') as output:
                        while block := response.read(256 * 1024):
                            output.write(block)
                if partial.stat().st_size != end - start + 1:
                    raise ValueError('Неполный диапазон')
                partial.replace(path)
                return path
            except Exception:
                if attempt == 11:
                    raise
                time.sleep(min(attempt + 1, 5))

    print(f"Загрузка: {item['destination']} ({size / 1048576:.1f} MiB)", flush=True)
    with ThreadPoolExecutor(max_workers=24) as pool:
        paths = []
        for index, path in enumerate(pool.map(part, spans), 1):
            paths.append(path)
            if index % 32 == 0:
                print(f'  Получено {index}/{len(spans)} диапазонов', flush=True)
    temporary = target.with_name(target.name + '.part')
    with temporary.open('wb') as out:
        for path in paths:
            with path.open('rb') as source:
                shutil.copyfileobj(source, out, 1024 * 1024)
    if temporary.stat().st_size != size:
        raise ValueError('Размер загрузки не совпадает')
    if item.get('sha256') or item.get('git_blob_sha1'):
        if not valid_file(temporary, {**item, 'size': size}):
            raise ValueError('Контрольная сумма загрузки не совпадает')
    temporary.replace(target)
    for path in paths:
        path.unlink()
    return target


def install(model, catalog, runtime, allow_unpinned=False):
    definition = catalog['models'][model]
    records = []
    for key in definition['artifacts']:
        item = catalog['artifacts'][key]
        if not (item.get('sha256') or item.get('git_blob_sha1')) and not allow_unpinned:
            raise ValueError('В каталоге не закреплён SHA: ' + key)
        path = download(item, runtime)
        record = {**item, 'actual_size': path.stat().st_size, 'actual_sha256': digest_file(path)}
        records.append(record)
        if item.get('archive_root'):
            models = (runtime / 'models').resolve()
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    parts = Path(member.filename.replace('\\', '/')).parts
                    if not parts or parts[0] != item['archive_root']:
                        raise ValueError('Неожиданный корень ZIP')
                    destination = (models / member.filename).resolve()
                    if not destination.is_relative_to(models):
                        raise ValueError('Путь архива выходит из папки моделей')
                    if member.is_dir():
                        destination.mkdir(parents=True, exist_ok=True)
                    else:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(member) as source, destination.open('wb') as out:
                            shutil.copyfileobj(source, out, 1024 * 1024)
        print('Проверено: ' + item['destination'], flush=True)
    manifests = runtime / 'manifests'
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / (model + '.json')).write_text(json.dumps({'model_id': model, 'definition': definition,
        'artifacts': records}, ensure_ascii=False, indent=2), encoding='utf-8')
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='whisper-small')
    parser.add_argument('--runtime', type=Path, default=ROOT / '.runtime')
    parser.add_argument('--allow-unpinned', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    catalog = json.loads((ROOT / 'configs/asr-models.json').read_text(encoding='utf-8'))
    models = list(catalog['models']) if args.model == 'all' else [args.model]
    if any(model not in catalog['models'] for model in models):
        parser.error('Неизвестная модель')
    for model in models:
        install(model, catalog, args.runtime, args.allow_unpinned)


if __name__ == '__main__':
    main()
