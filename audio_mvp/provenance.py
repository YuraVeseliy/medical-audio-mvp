"""Версия фактических исходников, включая ещё не закоммиченные изменения."""
import hashlib
from pathlib import Path
import subprocess
import platform


def hardware():
    import psutil
    cpu = platform.processor()
    if platform.system() == 'Windows':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                cpu = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        except OSError:
            pass
    return {'cpu': cpu, 'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(), 'ram_bytes': psutil.virtual_memory().total}


def source_version(root=None):
    root = Path(root or Path(__file__).resolve().parents[1])
    files = sorted((root / 'audio_mvp').glob('*.py')) + sorted((root / 'tests').glob('*.py'))
    files += [root / name for name in ('run_app.pyw', 'scripts/check.py', 'scripts/compare_asr.py',
              'scripts/check_gui_scenario.py', 'configs/asr-models.json', 'requirements.txt', 'requirements-lock.txt')
              if (root / name).is_file()]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {'commit': commit, 'source_sha256': digest.hexdigest(),
            'hashed_files': [p.relative_to(root).as_posix() for p in files]}
