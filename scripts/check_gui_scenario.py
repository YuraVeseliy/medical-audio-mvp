"""Инструментальный сценарий настоящего Tkinter; не личный ручной тест студента."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import tkinter as tk
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audio_mvp.gui import AudioApp
from audio_mvp.model import Segment
from audio_mvp.providers import ASRRecognizer, MODEL_BY_ID
from audio_mvp.provenance import source_version
from audio_mvp.session import Session


class ExternalRecognizer:
    model_id = 'external-test-asr'
    def transcribe(self, path):
        time.sleep(0.15)
        return [Segment(0, 1, 'Что вас беспокоит?'), Segment(1, 2, 'Болит левое плечо.')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', type=Path, default=ROOT / 'data/audio/dev01.wav')
    parser.add_argument('--runtime', type=Path, default=ROOT / '.runtime')
    parser.add_argument('--model', choices=list(MODEL_BY_ID), default=None)
    parser.add_argument('--out', type=Path, default=ROOT / 'reports/asr_gui')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    root = tk.Tk()
    root.withdraw()
    if args.model:
        app = AudioApp(root, project_root=ROOT, runtime_root=args.runtime)
        app.model_name.set(MODEL_BY_ID[args.model].label)
    else:
        app = AudioApp(root, session=Session(ExternalRecognizer()), project_root=ROOT)
    session = app.session
    errors, heartbeats = [], 0
    saved, exported = args.out / 'session.json', args.out / 'result.txt'
    try:
        with patch('audio_mvp.gui.messagebox.showerror', side_effect=lambda *a, **k: errors.append(a)), \
             patch('audio_mvp.gui.filedialog.askopenfilename', return_value=str(args.audio.resolve())):
            app.import_button.invoke()
            app.process_button.invoke()
            deadline = time.monotonic() + 240
            while app._busy and time.monotonic() < deadline:
                root.update()
                heartbeats += 1
                time.sleep(0.01)
        assert not app._busy, 'Операция не завершилась вовремя'
        assert not errors, errors
        original = session.document.transcript
        if args.model:
            assert session.document.model_id == args.model
        assert original and app.transcript.get('1.0', 'end-1c') == original
        assert len(app.model_combo['values']) == 8
        edited = original + '\nПроверка сохранения: тестовая правка.'
        app.editor.delete('1.0', 'end')
        app.editor.insert('1.0', edited)
        root.update()
        assert session.document.text == edited
        assert session.document.transcript == original
        with patch('audio_mvp.gui.filedialog.asksaveasfilename', return_value=str(saved)):
            app.save_button.invoke()
        with patch('audio_mvp.gui.filedialog.asksaveasfilename', return_value=str(exported)):
            app.export_button.invoke()
        app.new_button.invoke()
        assert session.document is None
        with patch('audio_mvp.gui.filedialog.askopenfilename', return_value=str(saved)):
            app.open_button.invoke()
        assert app.editor.get('1.0', 'end-1c') == edited
        assert app.transcript.get('1.0', 'end-1c') == original
        assert exported.read_text(encoding='utf-8') == edited + '\n'
        result = {'time_utc': datetime.now(timezone.utc).isoformat(), **source_version(ROOT),
                  'scenario': 'GUI import -> full ASR -> edit -> JSON/TXT -> new -> open',
                  'real_model': bool(args.model), 'model_id': args.model or 'external-test-asr',
                  'audio': str(args.audio), 'heartbeats_during_processing': heartbeats,
                  'successful': True, 'executor': 'agent instrumented Tkinter scenario',
                  'not_student_manual_acceptance': True,
                  'boundaries_replaced': ['OS file chooser', 'error message UI'] + ([] if args.model else ['external ASR']),
                  'document_metrics': session.document.metrics}
        (args.out / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'successful': True, 'real_model': bool(args.model), 'heartbeats': heartbeats,
                          'out': str(args.out)}, ensure_ascii=True))
        return 0
    finally:
        root.destroy()
        if hasattr(session.recognizer, 'close'):
            session.recognizer.close()


if __name__ == '__main__':
    raise SystemExit(main())
