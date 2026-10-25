"""Один инструментальный сценарий GUI; не личная приёмка студентом/врачом.

Настоящие Tk, Session, storage, файловый декодер и event loop. Только внешние
ASR/LLM, запуск сервера и системные диалоги заменены контролируемыми границами.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import tkinter as tk
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audio_mvp.gui import AudioApp
from audio_mvp.model import Evidence, Segment
from audio_mvp.session import Session


class Recognizer:
    def transcribe(self, path):
        time.sleep(0.35)
        return [Segment(0, 1, "Болит колено. Началось вчера.")]


class Extractor:
    def extract(self, text):
        return [Evidence("complaints", "Болит колено.", 0, 13),
                Evidence("anamnesis", "Началось вчера.", 14, 29)], []


def main():
    target = ROOT / "reports/gui-scenario"
    target.mkdir(parents=True, exist_ok=True)
    root = tk.Tk()
    root.withdraw()
    app = AudioApp(root, project_root=ROOT, session=Session(Recognizer(), Extractor()),
                   runtime=SimpleNamespace(base_url="http://127.0.0.1:8081",
                                           ensure_server=lambda: None, shutdown=lambda: None))
    steps = []
    beats = []
    def beat():
        beats.append(time.perf_counter())
        if not app._closing:
            root.after(20, beat)
    try:
        with patch("audio_mvp.gui.filedialog.askopenfilename", return_value=str(ROOT / "data/audio/dev01.wav")):
            app.import_audio()
        assert app.audio_path.name == "dev01.wav"
        steps.append("Файл выбран через публичное действие import_audio")
        beat()
        app.process_audio()
        assert app._busy and app.process_button.instate(["disabled"])
        deadline = time.monotonic() + 5
        while app._busy and time.monotonic() < deadline:
            root.update()
            time.sleep(0.005)
        assert not app._busy and len(beats) >= 5
        assert app.transcript.get("1.0", "end-1c") == "Болит колено. Началось вчера."
        steps.append(f"Обработка завершена; {len(beats)} событий heartbeat, конфликтующая кнопка заблокирована")
        row = app.evidence.get_children()[0]
        app.evidence.selection_set(row)
        app.show_evidence()
        assert app.transcript.tag_ranges("evidence")
        steps.append("Выбор цитаты выделяет источник")
        app.editors["complaints"].delete("1.0", "end")
        app.editors["complaints"].insert("1.0", "Исправлено человеком: левое колено.")
        app._sync_edits()
        saved = target / "session.json"
        exported = target / "result.txt"
        with patch("audio_mvp.gui.filedialog.asksaveasfilename", return_value=str(saved)):
            assert app.save_session(True)
        with patch("audio_mvp.gui.filedialog.asksaveasfilename", return_value=str(exported)):
            app.export_current()
        assert "левое колено" in exported.read_text(encoding="utf-8")
        steps.append("Текущая правка сохранена в JSON и TXT с кириллицей")
        app.new_session()
        assert app.session.document is None
        with patch("audio_mvp.gui.filedialog.askopenfilename", return_value=str(saved)):
            app.open_session()
        assert "левое колено" in app.editors["complaints"].get("1.0", "end")
        steps.append("Новый сеанс и повторное открытие восстанавливают именно сохранённую правку")
        result = {"id": "S01", "status": "passed", "time_utc": datetime.now(timezone.utc).isoformat(),
                  "scope": __doc__, "steps": steps,
                  "limitations": "Automated Tk state scenario; no visual QA, real ML, driver, microphone or human acceptance"}
        (target / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        app._closing = True
        root.destroy()


if __name__ == "__main__":
    main()
