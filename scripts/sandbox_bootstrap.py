"""Запуск pip/ensurepip в ограниченной Windows-среде агента.

Python 3.12 mkdir(0700) создаёт ACL, несовместимый с токеном этой среды.
Используем наследуемые права только для создаваемых временных каталогов
данного процесса. На обычном ПК этот вспомогательный запуск не требуется.
"""
import os
import runpy
import sys

original_mkdir = os.mkdir


def inherited_mkdir(path, mode=0o777, *, dir_fd=None):
    if os.name == "nt" and mode == 0o700:
        mode = 0o777
    return original_mkdir(path, mode, dir_fd=dir_fd)


if __name__ == "__main__":
    os.mkdir = inherited_mkdir
    module = sys.argv.pop(1)
    runpy.run_module(module, run_name="__main__")
