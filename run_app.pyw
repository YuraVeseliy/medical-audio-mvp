"""Открытие приложения двойным щелчком при настроенной ассоциации Python."""
import sys
from pathlib import Path


def entry():
    if "--cli" in sys.argv[1:]:
        from audio_mvp.cli import main as cli_main
        arguments = list(sys.argv[1:])
        arguments.remove("--cli")
        return cli_main(arguments)
    from audio_mvp.gui import main
    main()
    return 0

if __name__ == "__main__":
    # В оконной сборке стандартные потоки могут быть None. Библиотекам и CLI
    # нужен настоящий поток; журнал также делает отказ запуска проверяемым.
    if getattr(sys, "frozen", False):
        folder = Path(sys.executable).resolve().parent / ".runtime" / "logs"
        folder.mkdir(parents=True, exist_ok=True)
        stream = (folder / "application.log").open("a", encoding="utf-8", buffering=1)
        if sys.stdout is None:
            sys.stdout = stream
        if sys.stderr is None:
            sys.stderr = stream
        print("Запуск MedicalAudio; CLI=" + str("--cli" in sys.argv[1:]), flush=True)
    try:
        raise SystemExit(entry())
    except Exception:
        import traceback
        traceback.print_exc()
        if "--cli" not in sys.argv[1:]:
            from tkinter import messagebox
            messagebox.showerror("Не удалось запустить приложение",
                                 "Сохраните журнал .runtime/logs/application.log и версию комплекта.")
        raise SystemExit(1)
